class KeyFrameFinder:
    def __init__(
        self, min_inlier_ratio=0.5, min_inlier_count=100, max_frames_since_last=20
    ):
        """
        Initialize KeyFrameFinder to track keyframes based on inlier statistics.

        Args:
            min_inlier_ratio: Minimum ratio of inliers to consider a frame as tracking well
            min_inlier_count: Minimum absolute number of inliers required
            max_frames_since_last: Maximum frames allowed since last keyframe
        """
        self.keyframe_indices = []
        self.keyframe_inlier_counts = []
        self.last_frame_idx = -1
        self.min_inlier_ratio = min_inlier_ratio
        self.min_inlier_count = min_inlier_count
        self.max_frames_since_last = max_frames_since_last
        self.last_keyframe_inliers = None

    def needsKeyframe(self, frame_idx, inlier_count, total_points=None):
        """
        Check if a new frame should be a keyframe based on inlier statistics.

        Args:
            frame_idx: Current frame index
            inlier_count: Number of inliers matched with previous frame
            total_points: Total number of tracked points (if None, uses last keyframe inliers)

        Returns:
            Boolean indicating if this frame should be a keyframe
        """
        # Always make first frame a keyframe
        if len(self.keyframe_indices) == 0:
            self.keyframe_indices.append(frame_idx)
            self.last_keyframe_inliers = inlier_count
            self.last_frame_idx = frame_idx
            return True
        self.keyframe_inlier_counts.append(inlier_count)
        # Calculate inlier ratio
        reference_count = (
            self.last_keyframe_inliers if total_points is None else total_points
        )
        inlier_ratio = inlier_count / reference_count if reference_count > 0 else 0

        # Number of frames since last keyframe
        frames_since_last = frame_idx - self.keyframe_indices[-1]

        # Decide if we need a new keyframe
        is_keyframe = False

        # If inlier ratio is too low or absolute count is too low, create keyframe
        if (
            inlier_ratio < self.min_inlier_ratio
            or inlier_count < self.min_inlier_count
            or frames_since_last >= self.max_frames_since_last
        ):
            is_keyframe = True

        # Update state if this is a keyframe
        if is_keyframe:
            self.keyframe_indices.append(frame_idx)
            self.last_keyframe_inliers = inlier_count

        self.last_frame_idx = frame_idx
        return is_keyframe


# def select_keyframes(video_frames, method="uniform", threshold=0.3, max_gap=30):
#     """
#     Select keyframes from a video sequence.

#     Args:
#         video_frames: Tensor of shape [seq_len, 3, H, W]
#         method: Keyframe selection method
#             - "uniform": Select frames at regular intervals
#             - "difference": Select based on frame differences exceeding a threshold
#             - "optical_flow": Select based on optical flow magnitude
#             - "feature_change": Select based on feature embedding changes
#         threshold: Threshold value for non-uniform selection methods
#         max_gap: Maximum number of frames between keyframes

#     Returns:
#         List of keyframe indices
#     """
#     seq_len = video_frames.shape[0]
#     device = video_frames.device

#     # Always include the first frame
#     keyframe_indices = [0]

#     if method == "uniform":
#         # Simple uniform sampling of frames
#         step = max(1, min(int(seq_len / 10), max_gap))
#         keyframe_indices.extend(list(range(step, seq_len, step)))

#     elif method == "difference":
#         # Select frames based on image differences
#         last_keyframe = video_frames[0]
#         for i in range(1, seq_len):
#             current_frame = video_frames[i]

#             # Compute normalized frame difference
#             diff = torch.mean(torch.abs(current_frame - last_keyframe))
#             normalized_diff = diff / torch.mean(torch.abs(last_keyframe) + 1e-6)

#             # Check if difference exceeds threshold or max_gap reached
#             frames_since_last = i - keyframe_indices[-1]
#             if normalized_diff > threshold or frames_since_last >= max_gap:
#                 keyframe_indices.append(i)
#                 last_keyframe = current_frame

#     elif method == "optical_flow":
#         try:
#             import cv2
#         except ImportError:
#             print("OpenCV not available, falling back to difference method")
#             return select_keyframes(
#                 video_frames, method="difference", threshold=threshold, max_gap=max_gap
#             )

#         # Convert frames to numpy for OpenCV
#         frames_np = video_frames.cpu().numpy().transpose(0, 2, 3, 1)
#         if frames_np.shape[-1] == 3:  # Convert to grayscale if RGB
#             frames_gray = np.array(
#                 [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames_np]
#             )
#         else:
#             frames_gray = frames_np.squeeze(-1)

#         # Parameters for Lucas-Kanade optical flow
#         lk_params = dict(
#             winSize=(15, 15),
#             maxLevel=2,
#             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
#         )

#         # Feature params for goodFeaturesToTrack
#         feature_params = dict(
#             maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
#         )

#         # Get initial features to track
#         prev_frame = frames_gray[0]
#         prev_pts = cv2.goodFeaturesToTrack(prev_frame, mask=None, **feature_params)

#         for i in range(1, seq_len):
#             curr_frame = frames_gray[i]

#             if prev_pts is not None and len(prev_pts) > 0:
#                 # Calculate optical flow
#                 curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
#                     prev_frame, curr_frame, prev_pts, None, **lk_params
#                 )

#                 # Select good points
#                 good_old = prev_pts[status == 1]
#                 good_new = curr_pts[status == 1]

#                 if len(good_old) > 0 and len(good_new) > 0:
#                     # Calculate average flow magnitude
#                     flow_diffs = good_new - good_old
#                     avg_flow = np.mean(
#                         np.sqrt(flow_diffs[:, 0] ** 2 + flow_diffs[:, 1] ** 2)
#                     )

#                     # Check if flow exceeds threshold or max_gap reached
#                     frames_since_last = i - keyframe_indices[-1]
#                     if (
#                         avg_flow > threshold * prev_frame.shape[0]
#                         or frames_since_last >= max_gap
#                     ):
#                         keyframe_indices.append(i)

#                         # Reset tracking points at keyframes
#                         prev_pts = cv2.goodFeaturesToTrack(
#                             curr_frame, mask=None, **feature_params
#                         )
#                 else:
#                     # If tracking is lost, create a new keyframe
#                     keyframe_indices.append(i)
#                     prev_pts = cv2.goodFeaturesToTrack(
#                         curr_frame, mask=None, **feature_params
#                     )
#             else:
#                 # If no points to track, create a new keyframe
#                 keyframe_indices.append(i)
#                 prev_pts = cv2.goodFeaturesToTrack(
#                     curr_frame, mask=None, **feature_params
#                 )

#             prev_frame = curr_frame

#     elif method == "feature_change":
#         # This requires a feature extractor model
#         # For simplicity, we'll use a basic feature representation using image pyramids
#         from torch.nn.functional import avg_pool2d

#         # Create a simple feature pyramid
#         def extract_features(frame):
#             features = []
#             x = frame
#             for _ in range(3):  # 3 pyramid levels
#                 features.append(x)
#                 x = avg_pool2d(x, kernel_size=2)
#             return torch.cat(
#                 [F.interpolate(f, size=frame.shape[-2:]) for f in features], dim=1
#             )

#         # Extract features for first frame
#         last_features = extract_features(video_frames[0].unsqueeze(0)).squeeze(0)

#         for i in range(1, seq_len):
#             # Extract features for current frame
#             current_features = extract_features(video_frames[i].unsqueeze(0)).squeeze(0)

#             # Compute normalized feature difference
#             feature_diff = torch.mean(torch.abs(current_features - last_features))
#             normalized_diff = feature_diff / (
#                 torch.mean(torch.abs(last_features)) + 1e-6
#             )

#             # Check if difference exceeds threshold or max_gap reached
#             frames_since_last = i - keyframe_indices[-1]
#             if normalized_diff > threshold or frames_since_last >= max_gap:
#                 keyframe_indices.append(i)
#                 last_features = current_features

#     # Always include the last frame if not already included
#     if keyframe_indices[-1] != seq_len - 1:
#         keyframe_indices.append(seq_len - 1)

#     return keyframe_indices


# def compute_frame_importance(frame1, frame2, method="difference"):
#     """
#     Compute the importance/information content between two frames.

#     Args:
#         frame1: First frame tensor [3, H, W]
#         frame2: Second frame tensor [3, H, W]
#         method: Method to compute importance

#     Returns:
#         Scalar importance value
#     """
#     if method == "difference":
#         # Simple pixel-wise difference
#         diff = torch.mean(torch.abs(frame2 - frame1))
#         return diff / (torch.mean(torch.abs(frame1)) + 1e-6)

#     elif method == "ssim":
#         # Structural similarity (higher SSIM means more similar)
#         from torch.nn.functional import conv2d, pad

#         def gaussian_kernel(size, sigma):
#             coords = torch.arange(size, dtype=torch.float32)
#             coords -= size // 2
#             g = coords**2
#             g = torch.exp(-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma**2))
#             g /= g.sum()
#             return g

#         kernel_size = 11
#         sigma = 1.5
#         kernel = gaussian_kernel(kernel_size, sigma).to(frame1.device)
#         kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)

#         c1 = 0.01**2
#         c2 = 0.03**2

#         mu1 = conv2d(frame1.unsqueeze(0), kernel, padding=kernel_size // 2)
#         mu2 = conv2d(frame2.unsqueeze(0), kernel, padding=kernel_size // 2)

#         mu1_sq = mu1**2
#         mu2_sq = mu2**2
#         mu1_mu2 = mu1 * mu2

#         sigma1_sq = (
#             conv2d(frame1.unsqueeze(0) ** 2, kernel, padding=kernel_size // 2) - mu1_sq
#         )
#         sigma2_sq = (
#             conv2d(frame2.unsqueeze(0) ** 2, kernel, padding=kernel_size // 2) - mu2_sq
#         )
#         sigma12 = (
#             conv2d(
#                 frame1.unsqueeze(0) * frame2.unsqueeze(0),
#                 kernel,
#                 padding=kernel_size // 2,
#             )
#             - mu1_mu2
#         )

#         ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
#             (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
#         )
#         ssim_value = torch.mean(ssim_map)

#         # Return 1 - SSIM as the importance (higher value means more different)
#         return 1.0 - ssim_value

#     else:
#         raise ValueError(f"Unknown importance computation method: {method}")


# def filter_redundant_keyframes(video_frames, keyframe_indices, threshold=0.1):
#     """
#     Remove redundant keyframes that are too similar to adjacent ones.

#     Args:
#         video_frames: Tensor of shape [seq_len, 3, H, W]
#         keyframe_indices: List of current keyframe indices
#         threshold: Redundancy threshold

#     Returns:
#         Filtered list of keyframe indices
#     """
#     if len(keyframe_indices) <= 2:
#         # Always keep first and last frame
#         return keyframe_indices

#     filtered_indices = [keyframe_indices[0]]  # Always keep the first frame

#     for i in range(1, len(keyframe_indices) - 1):
#         prev_idx = keyframe_indices[i - 1]
#         curr_idx = keyframe_indices[i]
#         next_idx = keyframe_indices[i + 1]

#         # Compute importance with previous and next frames
#         prev_importance = compute_frame_importance(
#             video_frames[prev_idx], video_frames[curr_idx]
#         )
#         next_importance = compute_frame_importance(
#             video_frames[curr_idx], video_frames[next_idx]
#         )

#         # Keep the frame if it's significantly different from neighbors
#         if prev_importance > threshold or next_importance > threshold:
#             filtered_indices.append(curr_idx)

#     filtered_indices.append(keyframe_indices[-1])  # Always keep the last frame

#     return filtered_indices
