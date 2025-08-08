from turtle import update
import numpy as np
import torch
import rerun as rr
import rerun.blueprint as rrb

import pipelines.matching as matching
import pipelines.odometry as odometry
from utilities import visualization
from utilities.visualization import log_to_rerun, log_rerun_line
from scipy.spatial.transform import Rotation

torch.autograd.set_detect_anomaly(True)

class OdometryPipeline:
    def __init__(
        self,
        config,
        model=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        tqdm=False,
        to_rerun=False,
    ):
        ### INITIALIZATIONS
        self.config = config
        self.device = device
        self.use_tqdm = tqdm
        self.to_rerun = to_rerun
        if to_rerun:
            self.rerun_init()

        ### ODOMETRY PIPELINE INITIALIZATIONS
        # The matching pipeline is used to extract features and compute correspondences
        self.match = matching.matching.MatchingPipeline(
            config, model=model, device=self.device
        )
        self.trajectory = odometry.trajectory.Trajectory(
            device=self.device,
            min_inlier_ratio=0.1,
            min_inlier_count=80,
            max_frames_since_last=20,
        )
        # Ground truth trajectory uses same parameters but won't need inlier checks
        self.gt_trajectory = odometry.trajectory.Trajectory(
            device=self.device,
            min_inlier_ratio=0.0,  # Don't use inlier ratio for ground truth
            min_inlier_count=0,  # Don't use inlier count for ground truth
            max_frames_since_last=float(
                "inf"
            ),  # Don't force keyframes for ground truth
        )

        ### DIMENSIONS
        self.height, self.width, self.batch_size = (
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
            config.BATCH_SIZE,
        )

        ### METRICS
        self.metrics = None
        self.inlier_counts = []
        self.warp = matching.projections.Warp(384, 384).cuda()

    def initialize(self):
        self.trajectory.initialize()
        
    def process_dataset(self, dataset, batch_size=1, num_workers=1, compute_gt=False):
        """
        Process a dataset to extract camera trajectory.

        Args:
            dataset: Dataset object containing video frames
            batch_size: Batch size for the dataloader
            num_workers: Number of workers for the dataloader
            compute_gt: Whether to compute ground truth trajectory

        Returns:
            Dictionary containing trajectory and other relevant information
        """
        # Create dataloader from dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=dataset.sampler,
            pin_memory=True,
        )

        ### INITIALIZATION
        # Instantiate trajectories
        n_frames = len(dataloader)
        self.trajectory.initialize(num_frames=n_frames)
        prev_pose = self.trajectory.trajectory[0, :3, 3].cpu()

        # Initialize ground truth trajectory if needed
        if compute_gt:
            self.gt_trajectory.initialize(n_frames, base_pose=dataset[0]["Ts"])
            prev_pose_gt = self.gt_trajectory.trajectory[0, :3, 3].cpu()
        else:
            prev_pose_gt = None

        # Wrap dataloader with tqdm if enabled
        if self.use_tqdm:
            from tqdm import tqdm

            dataloader = tqdm(
                dataloader, total=len(dataloader), desc="Processing frames"
            )

        ### LOOP THROUGH CONSECUTIVE FRAME PAIRS
        for frame_idx, sample in enumerate(dataloader):
            ### DATA LOADING
            framestack = sample["framestack"].to(self.device)
            current_frame = framestack[:, 1]
            K = sample["intrinsics"].to(self.device)

            # Initialize keyframes at frame 0
            if frame_idx == 0:
                self.trajectory.update_keyframe(current_frame, frame_idx)
                self.gt_trajectory.update_keyframe(current_frame, frame_idx)

            # Retrieve global poses if compute_gt is True
            if compute_gt:
                Tt = sample["Tt"].to(self.device)  # Current frame global pose
                Ts = sample["Ts"].to(self.device)  # Source frame global pose

            # Compute the depthmap of the keyframe to backproject to the point cloud
            if not hasattr(
                self, "cached_keyframe_depth"
            ) or self.trajectory.keyframe_idx != getattr(
                self, "cached_keyframe_idx", -1
            ):
                if "depthstack" in sample.keys():
                    source_depth = sample["depthstack"][:, 0].to(self.device)
                else:
                    source_depth = self.match.model.depth(self.trajectory.keyframe)

                # Cache the depth and keyframe index
                self.cached_keyframe_depth = source_depth
                self.cached_keyframe_idx = self.trajectory.keyframe_idx
            else:
                # Use cached depth if keyframe hasn't changed
                source_depth = self.cached_keyframe_depth

            ### COMPUTE PIXEL CORRESPONDENCES - Matching pipeline
            matchout = self.match.match_images(
                image1=self.trajectory.keyframe, image2=current_frame
            )
            source_pixels_matched = matchout["source_pixels_matched"]
            target_pixels_matched = matchout["target_pixels_matched"]
            batch_idx_match = matchout["batch_idx_match"]
            inliers = matchout["inliers"]
            scores = matchout["scores"]

            ### PERSPECTIVE-N-POINT ALGORITHM - COMPUTE POSE FROM KEYFRAME
            # Localize the current frame with respect to the keyframe cloud
            depthmap = 1/ source_depth * self.config.DEPTH_SCALE_FACTOR + self.config.DEPTH_BIAS_FACTOR
            relative_pose = self.trajectory.estimate_relative_pose_pnp(
                source_pixels_matched,
                target_pixels_matched,
                batch_idx_match,
                K,
                depthmap,
            )
            if compute_gt:
                relative_pose_gt = Ts @ torch.inverse(Tt)
                relative_pose_translation_scale = torch.norm(relative_pose[0, :3, 3])
                relative_pose_translation_scale_gt = torch.norm(
                    relative_pose_gt[0, :3, 3]
                )

                # ! Just for visualization / evaluation purposes, we scale the ground truth relative pose to match the estimated scale
                relative_pose_gt[0, :3, 3] *= (
                    relative_pose_translation_scale / relative_pose_translation_scale_gt
                )

            ### TRAJECTORY COMPUTATION - LOCALIZE CURRENT FRAME AS GLOBAL POSE
            self.trajectory.update_from_keyframe(
                self.trajectory.keyframe_idx, frame_idx, relative_pose[0]
            )
            if compute_gt:
                self.gt_trajectory.update_from_keyframe(
                    self.gt_trajectory.keyframe_idx, frame_idx, relative_pose_gt[0]
                )

            ### KEYFRAME UPDATE CHECK
            needs_keyframe = self.trajectory.needs_keyframe(frame_idx, inliers.sum())
            self.inlier_counts.append(inliers.sum())
            # IF we need a keyframe because of a low inlier count, we will use the current frame as the global pose RF
            if needs_keyframe:
                self.trajectory.update_keyframe(current_frame, frame_idx)
                if compute_gt:
                    self.gt_trajectory.update_keyframe(current_frame, frame_idx)
            if self.to_rerun:
                # Pass the keyframe image if this is a keyframe
                rgb_image = current_frame
                self.log_to_rerun(
                    frame_idx,
                    sample,
                    K,
                    depthmap,
                    prev_pose,
                    frame_idx == 0,
                    prev_pose_gt if compute_gt else None,
                    rgb_image,
                )

            prev_pose = self.trajectory.trajectory[frame_idx, :3, 3].cpu()
            if compute_gt:
                prev_pose_gt = self.gt_trajectory.trajectory[frame_idx, :3, 3].cpu()

        result = {
            "trajectory": self.trajectory.trajectory,
            "keyframe_idxs": self.trajectory.keyframer.keyframe_indices,
            "keyframe_inliers": self.trajectory.keyframer.keyframe_inlier_counts,
        }

        if compute_gt:
            result["gt_trajectory"] = self.gt_trajectory.trajectory
            result["gt_keyframe_idxs"] = self.gt_trajectory.keyframer.keyframe_indices
        self.inlier_counts = torch.tensor(self.inlier_counts)
        return result

    def save_if_keyframe(self, is_keyframe, framestack, idx):
        """
        Save the current frame as a keyframe if it is a keyframe.
        """
        if is_keyframe:
            self.trajectory.update_keyframe(framestack[:, 0], idx)

    def estimate_intrinsics(self, frames):
        """
        Estimate camera intrinsics if not provided.

        Args:
            frames: Tensor of shape [B, 2, 3, H, W]

        Returns:
            Camera intrinsic matrix
        """
        # Simple default intrinsics based on image dimensions
        h, w = frames.shape[-2], frames.shape[-1]
        focal = max(h, w) * 0.8
        K = torch.tensor(
            [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], device=frames.device
        )
        return K

    def compute_metrics(self, estimated_trajectory=None, gt_trajectory=None):
        """
        Compute comprehensive trajectory evaluation metrics.

        This method computes various metrics to evaluate the quality of the estimated trajectory,
        including position errors, rotation errors, relative pose errors, and trajectory smoothness.
        If ground truth is provided, it also computes comparison metrics.

        Args:
            estimated_trajectory: Tensor of shape (F, 4, 4) representing estimated camera poses (optional)
            gt_trajectory: Tensor of shape (F, 4, 4) representing ground truth camera poses (optional)

        Returns:
            Dictionary containing all computed metrics and statistics
        """
        # Use pipeline trajectories if none provided
        if estimated_trajectory is None:
            estimated_trajectory = self.trajectory.trajectory
        if gt_trajectory is None and hasattr(self, "gt_trajectory"):
            gt_trajectory = self.gt_trajectory.trajectory

        metrics = {}

        # Compute trajectory smoothness (always computed)
        metrics["smoothness"] = odometry.metrics.trajectory_smoothness(
            estimated_trajectory
        )

        if gt_trajectory is not None:
            # Compute all trajectory errors
            errors = odometry.metrics.compute_trajectory_errors(
                estimated_trajectory, gt_trajectory
            )

            # Compute error statistics
            stats = odometry.metrics.compute_error_statistics(errors)

            # Store all errors and statistics
            metrics["errors"] = errors
            metrics["statistics"] = stats

            # Store key metrics for easy access
            metrics["ate"] = {
                "mean": stats["translation_errors_mean"].item(),
                "rmse": torch.sqrt(
                    torch.mean(errors["translation_errors"] ** 2)
                ).item(),
                "max": stats["translation_errors_max"].item(),
                "min": stats["translation_errors_min"].item(),
            }

            metrics["rpe"] = {
                "mean": stats["relative_pose_errors_mean"].item(),
                "rmse": torch.sqrt(
                    torch.mean(errors["relative_pose_errors"] ** 2)
                ).item(),
                "max": stats["relative_pose_errors_max"].item(),
                "min": stats["relative_pose_errors_min"].item(),
            }

            # Store drift metrics
            metrics["drift"] = {
                "per_distance": {
                    "mean": torch.mean(
                        errors["drift_per_distance"][errors["drift_per_distance"] > 0]
                    ).item(),
                    "max": torch.max(errors["drift_per_distance"]).item(),
                },
                "per_time": {
                    "mean": torch.mean(
                        errors["drift_per_time"][errors["drift_per_time"] > 0]
                    ).item(),
                    "max": torch.max(errors["drift_per_time"]).item(),
                },
            }

            # Store cumulative errors
            metrics["cumulative"] = {
                "translation": stats["final_cum_translation_error"].item(),
                "rotation": stats["final_cum_rotation_error"].item(),
            }

        # Store metrics as attribute
        self.metrics = metrics
        return metrics

    def visualize_metrics(self, save_dir=None):
        """
        Generate visualizations and summary of trajectory metrics.

        This method creates comprehensive visualizations of trajectory errors and metrics,
        including position errors, rotation errors, drift analysis, and 3D trajectory comparison.
        It also generates a summary table of key metrics.

        Args:
            save_dir: Directory to save the visualizations (optional)

        Returns:
            None
        """
        if self.metrics is None:
            print("No metrics available. Run compute_metrics first.")
            return

        if "errors" not in self.metrics:
            print(
                "No error metrics available for visualization. Run compute_metrics with ground truth first."
            )
            return

        # Create save paths if directory is provided
        if save_dir is not None:
            import os

            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, "trajectory_errors.png")
        else:
            plot_path = None

        # Generate plots
        odometry.metrics.plot_trajectory_errors(
            errors=self.metrics["errors"],
            trajectory=self.trajectory.trajectory,
            trajectory_gt=(
                self.gt_trajectory.trajectory
                if hasattr(self, "gt_trajectory")
                else None
            ),
            save_path=plot_path,
        )

        # Print statistics
        odometry.metrics.print_error_statistics(self.metrics["statistics"])

        # Print key metrics in a readable format
        print("\n===== KEY METRICS SUMMARY =====")
        print(f"Absolute Trajectory Error (ATE):")
        print(f"  Mean: {self.metrics['ate']['mean']:.4f}")
        print(f"  RMSE: {self.metrics['ate']['rmse']:.4f}")
        print(f"  Max:  {self.metrics['ate']['max']:.4f}")

        print(f"\nRelative Pose Error (RPE):")
        print(f"  Mean: {self.metrics['rpe']['mean']:.4f}")
        print(f"  RMSE: {self.metrics['rpe']['rmse']:.4f}")
        print(f"  Max:  {self.metrics['rpe']['max']:.4f}")

        print(f"\nDrift Metrics:")
        print(
            f"  Per Distance - Mean: {self.metrics['drift']['per_distance']['mean']:.4f}"
        )
        print(f"  Per Time - Mean: {self.metrics['drift']['per_time']['mean']:.4f}")

        print(f"\nCumulative Errors:")
        print(f"  Translation: {self.metrics['cumulative']['translation']:.4f}")
        print(f"  Rotation: {self.metrics['cumulative']['rotation']:.4f}")

        print("\n=============================")

    def log_to_rerun(
        self,
        frame_idx,
        sample,
        K,
        source_depth,
        prev_pose,
        force_cloud=False,
        prev_pose_gt=None,
        rgb_image=None,
    ):
        """
        Log trajectory and visualization data to rerun.

        Args:
            frame_idx: Current frame index
            sample: Current sample containing frames
            K: Camera intrinsic matrix
            source_depth: Depth map of the keyframe
            prev_pose: Previous camera pose
            prev_pose_gt: Previous ground truth camera pose
            rgb_image: Optional RGB image to display on the camera frustum
        """
        if frame_idx == 0 or force_cloud:
            from pipelines.matching.projections import BackProject

            cloud, _, rgb_vec = (
                BackProject(
                    sample["framestack"].shape[-2], sample["framestack"].shape[-1]
                )
                .to(self.device)(
                    self.trajectory.keyframe, source_depth, torch.inverse(K)
                )
                .values()
            )
            # Transform cloud to world coordinates
            # cloud = cloud[0]
            cloud = self.trajectory.trajectory[frame_idx] @ cloud[0]
            log_to_rerun(cloud=cloud, rgb_vec=rgb_vec[0], frame=frame_idx, K=K[0])

        import losses

        photometric = losses.WeightedCombinationLoss(
            components=[
                ("SSIM", losses.SSIMLoss(), 0.8, ("target", "warped")),
                ("L1", losses.L1Loss(), 0.2, ("target", "warped")),
            ],
        )
        warped_keyframe = self.warp(
            self.trajectory.keyframe.cuda(),
            source_depth.cuda(),
            K.cuda(),
            torch.inverse(
                self.trajectory.trajectory[self.trajectory.keyframe_idx]
            )
            @ self.trajectory.trajectory[frame_idx].unsqueeze(0).cuda(),
        )["warped"][0]
        # Log metrics for estimated trajectory
        metrics_dict = {
            "x/pred": {
                "value": self.trajectory.trajectory[frame_idx, 0, 3].item(),
                "color": "#ff0000",
            },
            "y/pred": {
                "value": self.trajectory.trajectory[frame_idx, 1, 3].item(),
                "color": "#00ff00",
            },
            "z/pred": {
                "value": self.trajectory.trajectory[frame_idx, 2, 3].item(),
                "color": "#0000ff",
            },
            "inliers/count": {
                "value": self.inlier_counts[frame_idx].item(),
                "color": "#ffff00",
            },
            "inliers/flag": {
                "value": (
                    self.inlier_counts[0].item()
                    if frame_idx == self.trajectory.keyframe_idx
                    else 0
                ),
                "color": "#ffffff",
            },
            "photometric": {
                "value": photometric(
                    target=self.trajectory.keyframe,
                    warped=warped_keyframe,
                ).cpu(),  # , K, self.trajectory.trajectory[frame_idx]),
                "color": "#ffffff",
            },
        }

        # Add ground truth metrics if available
        if hasattr(self, "gt_trajectory") and self.gt_trajectory.trajectory is not None:
            metrics_dict.update(
                {
                    "x/gt": {
                        "value": self.gt_trajectory.trajectory[frame_idx, 0, 3].item(),
                        "color": "#ff9999",
                    },
                    "y/gt": {
                        "value": self.gt_trajectory.trajectory[frame_idx, 1, 3].item(),
                        "color": "#99ff99",
                    },
                    "z/gt": {
                        "value": self.gt_trajectory.trajectory[frame_idx, 2, 3].item(),
                        "color": "#9999ff",
                    },
                    "error/x": {
                        "value": abs(
                            self.gt_trajectory.trajectory[frame_idx, 0, 3].item()
                            - self.trajectory.trajectory[frame_idx, 0, 3].item()
                        ),
                        "color": "#ff9999",
                    },
                    "error/y": {
                        "value": abs(
                            self.gt_trajectory.trajectory[frame_idx, 1, 3].item()
                            - self.trajectory.trajectory[frame_idx, 1, 3].item()
                        ),
                        "color": "#99ff99",
                    },
                    "error/z": {
                        "value": abs(
                            self.gt_trajectory.trajectory[frame_idx, 2, 3].item()
                            - self.trajectory.trajectory[frame_idx, 2, 3].item()
                        ),
                        "color": "#9999ff",
                    },
                }
            )

        log_to_rerun(metrics=metrics_dict, frame=frame_idx)

        # Log estimated trajectory
        R = self.trajectory.trajectory[frame_idx, :3, :3].cpu()
        t = self.trajectory.trajectory[frame_idx, :3, 3].cpu()

        # Convert rotation matrix to axis-angle
        rot = Rotation.from_matrix(R)
        axis_angle = rot.as_rotvec()
        axis = axis_angle / (np.linalg.norm(axis_angle) + 1e-10)  # Normalize axis
        angle = np.linalg.norm(axis_angle)  # Extract angle

        rr.set_time_sequence("frame", frame_idx)

        # Log connection lines between consecutive estimated poses
        if frame_idx > 0:
            log_rerun_line(
                source=self.trajectory.trajectory[frame_idx, :3, 3].cpu(),
                target=prev_pose,
                entity=f"/cloud/{frame_idx-1}_to_{frame_idx}",
                colors=[1.0, 0.5, 0.0],  # Green-cyan
                radii=0.2,
            )
            if prev_pose_gt is not None:
                log_rerun_line(
                    source=self.gt_trajectory.trajectory[frame_idx, :3, 3].cpu(),
                    target=prev_pose_gt,
                    entity=f"/cloud/gt_{frame_idx-1}_to_{frame_idx}",
                    colors=[0.0, 1, 0.0],  # Green
                    radii=0.5,
                )

        # Log connection line to keyframe
        # log_rerun_line(
        #     source=self.trajectory.trajectory[frame_idx, :3, 3].cpu(),
        #     target=self.trajectory.trajectory[self.trajectory.keyframe_idx, :3, 3].cpu(),
        #     entity=f"/cloud/{frame_idx}_to_KF",
        #     colors=[1.0, 0.5, 0.0],  # Orange
        #     radii=0.05,
        # )

        # Log estimated camera pose
        rr.log(
            f"/cloud/{frame_idx}",
            rr.Transform3D(
                translation=t,
                rotation_axis_angle=rr.RotationAxisAngle(axis=axis, radians=angle),
                axis_length=0.25,
            ),
        )

        rr.log("/image/rgb", rr.Image(rgb_image[0].cpu().permute(1, 2, 0).numpy()))
        rr.log(
            "/image/warped",
            rr.Image(
                warped_keyframe.cpu().permute(1, 2, 0).numpy()
            ),
        )

        kfm = 2 if self.trajectory.keyframe_idx == frame_idx else 1
        rr.log(
            f"/cloud/{frame_idx}/cam",
            rr.Pinhole(
                width=sample["framestack"].shape[-2],
                height=sample["framestack"].shape[-1],
                focal_length=(
                    K[0, 0, 0].cpu(),
                    K[0, 1, 1].cpu(),
                ),
                principal_point=(
                    K[0, 0, 2].cpu(),
                    K[0, 1, 2].cpu(),
                ),
                image_plane_distance=2 * kfm,
            ),
        )

        # Log RGB image if provided
        if self.trajectory.keyframe_idx == frame_idx:
            rr.log(
                f"/cloud/{frame_idx}/cam/image",
                rr.Image(rgb_image[0].cpu().permute(1, 2, 0).numpy()),
            )

    def rerun_init(self):
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/cloud",
                    name="3D",
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        origin="/image/rgb",
                        name="RGB",
                    ),
                    rrb.Spatial2DView(
                        origin="/image/depth",
                        name="Depth",
                    ),
                    rrb.Spatial2DView(
                        origin="/image/raycast",
                        name="Raycast",
                    ),
                    row_shares=[0.3, 0.3, 0.3],
                ),
                # Make the time series view occupy 20% of the height
                column_shares=[0.7, 0.3],  # 80% for 3D view, 20% for time series
            ),
            collapse_panels=True,
        )
        rr.init("Trajectory Rec")
        rr.serve(ws_port=0, web_port=0)
        rr.send_blueprint(blueprint)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)
