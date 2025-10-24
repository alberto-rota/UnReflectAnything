import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
import sys

# Ensure project root on sys.path to import metrics.py reliably
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import metrics as ua_metrics


@dataclass
class MethodConfig:
    name: str
    output_dir_name: str
    enabled: bool = True


def read_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def is_image_file(path: Path, allowed_exts: List[str]) -> bool:
    return path.suffix.lower().lstrip(".") in {e.lower() for e in allowed_exts}


def list_subdatasets(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_images_in_subdataset(folder: Path, exts: List[str]) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and is_image_file(p, exts)])


def pil_to_torch_chw_uint8(img: Image.Image) -> torch.Tensor:
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    tensor = torch.from_numpy(arr)
    tensor = tensor.permute(2, 0, 1).contiguous()  # CHW
    return tensor


def load_image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return pil_to_torch_chw_uint8(img)


def locate_mask_for(stem: str, subdataset: str, cfg_masks: dict, benchmark_dir: Path) -> Optional[Path]:
    if not cfg_masks.get("enabled", False):
        return None
    base_dir = cfg_masks.get("base_dir")
    glob_pattern = cfg_masks.get("glob_pattern")
    if base_dir:
        candidate = Path(base_dir) / subdataset / f"{stem}.png"
        return candidate if candidate.exists() else None
    if glob_pattern:
        pat = glob_pattern.format(benchmark_dir=str(benchmark_dir), subdataset=subdataset, stem=stem)
        matches = list(Path(benchmark_dir).glob(pat))
        return matches[0] if matches else None
    return None


def ensure_dirs(paths: List[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


def build_manifest(cfg: dict) -> pd.DataFrame:
    benchmark_dir = Path(cfg["benchmark_dir"]).resolve()
    input_root = benchmark_dir / "input"
    gt_root = benchmark_dir / "diffuse_gt"

    methods_cfg = [MethodConfig(**m) for m in cfg["methods"] if m.get("enabled", True)]
    exts = cfg["images"]["extensions"]

    subdatasets = list_subdatasets(input_root)

    rows: List[Dict] = []
    for sub in subdatasets:
        input_sub = input_root / sub
        gt_sub = gt_root / sub
        input_images = list_images_in_subdataset(input_sub, exts)
        for inp_path in input_images:
            stem = inp_path.stem
            gt_path = (gt_sub / f"{stem}{inp_path.suffix}") if gt_sub.exists() else None
            gt_path = gt_path if (gt_path and gt_path.exists()) else None
            mask_path = locate_mask_for(stem, sub, cfg.get("masks", {}), benchmark_dir)
            has_gt = gt_path is not None
            has_mask = mask_path is not None
            for m in methods_cfg:
                pred_path = benchmark_dir / m.output_dir_name / sub / f"{stem}{inp_path.suffix}"
                has_pred = pred_path.exists()
                if not has_pred and cfg["policy"].get("skip_missing_pred", True):
                    continue
                rows.append(
                    {
                        "subdataset": sub,
                        "image_id": stem,
                        "global_image_key": f"{sub}/{stem}",
                        "input_path": str(inp_path),
                        "gt_path": str(gt_path) if gt_path else None,
                        "mask_path": str(mask_path) if has_mask else None,
                        "method": m.name,
                        "pred_path": str(pred_path) if has_pred else None,
                        "has_input": True,
                        "has_gt": has_gt,
                        "has_mask": has_mask,
                        "has_pred": has_pred,
                    }
                )

    manifest = pd.DataFrame(rows)
    manifest.sort_values(["subdataset", "image_id", "method"], inplace=True)
    return manifest


def batch(iterable: List, n: int):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def to_bchw_uint8(tensors: List[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 0:
        return torch.empty(0, 3, 0, 0, dtype=torch.uint8)
    return torch.stack(tensors, dim=0)  # B,C,H,W


def compute_metrics_for_batch(
    cfg: dict,
    batch_inputs: torch.Tensor,   # [B,3,H,W] uint8
    batch_preds: torch.Tensor,    # [B,3,H,W] uint8
    batch_gts: Optional[torch.Tensor],    # [B,3,H,W] uint8 (placeholders allowed)
    batch_masks: Optional[torch.Tensor],  # [B,1,H,W] float32 in {0,1} (placeholders allowed)
    have_gt_flags: List[bool],
    have_mask_flags: List[bool],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    data_range = cfg.get("data_range_override")

    B = batch_preds.shape[0]
    # Initialize all outputs as NaN (on device for assignment, move to cpu later)
    def nan_vec():
        return torch.full((B,), float("nan"), device=device, dtype=torch.float32)

    out: Dict[str, torch.Tensor] = {
        "mse": nan_vec(), "psnr": nan_vec(), "ssim": nan_vec(), "lpips_vgg": nan_vec(),
        "dists": nan_vec(), "gmsd": nan_vec(), "deltaE2000": nan_vec(),
        "brisque": nan_vec(), "niqe": nan_vec(), "piqe": nan_vec(),
        "boundary_gmsd_band3": nan_vec(), "luminance_suppression_ratio": nan_vec(),
        "chroma_consistency_deltaE_ring3": nan_vec(),
    }

    idx_all = torch.arange(B, device=device)
    idx_gt = idx_all[torch.tensor(have_gt_flags, device=device)] if any(have_gt_flags) else None
    idx_mask = idx_all[torch.tensor(have_mask_flags, device=device)] if any(have_mask_flags) else None
    idx_mask_gt = (
        idx_all[torch.tensor([g and m for g, m in zip(have_gt_flags, have_mask_flags)], device=device)]
        if any(have_gt_flags) and any(have_mask_flags)
        else None
    )

    # No-reference metrics for masked subset (with composite)
    if idx_mask is not None and idx_mask.numel() > 0:
        br = ua_metrics.brisque_metric(batch_preds[idx_mask], mask=batch_masks[idx_mask], reference_image_for_outside=batch_inputs[idx_mask], data_range=data_range, reduction="none")
        nq = ua_metrics.niqe_metric(batch_preds[idx_mask], mask=batch_masks[idx_mask], reference_image_for_outside=batch_inputs[idx_mask], data_range=data_range, reduction="none")
        pq = ua_metrics.piqe_metric(batch_preds[idx_mask], mask=batch_masks[idx_mask], reference_image_for_outside=batch_inputs[idx_mask], data_range=data_range, reduction="none")
        out["brisque"][idx_mask] = br
        out["niqe"][idx_mask] = nq
        out["piqe"][idx_mask] = pq

    # No-reference metrics for unmasked subset
    if idx_mask is None or idx_mask.numel() < B:
        mask_bool = torch.tensor(have_mask_flags, device=device)
        idx_nomask = idx_all[~mask_bool]
        if idx_nomask.numel() > 0:
            br = ua_metrics.brisque_metric(batch_preds[idx_nomask], mask=None, reference_image_for_outside=None, data_range=data_range, reduction="none")
            nq = ua_metrics.niqe_metric(batch_preds[idx_nomask], mask=None, reference_image_for_outside=None, data_range=data_range, reduction="none")
            pq = ua_metrics.piqe_metric(batch_preds[idx_nomask], mask=None, reference_image_for_outside=None, data_range=data_range, reduction="none")
            out["brisque"][idx_nomask] = br
            out["niqe"][idx_nomask] = nq
            out["piqe"][idx_nomask] = pq

    # Full-reference metrics for samples with GT
    if idx_gt is not None and idx_gt.numel() > 0 and batch_gts is not None:
        pr = batch_preds[idx_gt]
        gt = batch_gts[idx_gt]
        out["mse"][idx_gt] = ua_metrics.mse_metric(pr, gt, mask=None, reduction="none")
        out["psnr"][idx_gt] = ua_metrics.psnr_metric(pr, gt, mask=None, data_range=data_range, reduction="none")
        out["ssim"][idx_gt] = ua_metrics.ssim_metric(pr, gt, mask=None, data_range=data_range, reduction="none")
        out["lpips_vgg"][idx_gt] = ua_metrics.lpips_metric(pr, gt, mask=None, net="vgg", reduction="none", data_range=data_range)
        out["dists"][idx_gt] = ua_metrics.dists_metric(pr, gt, mask=None, reduction="none", data_range=data_range)
        out["gmsd"][idx_gt] = ua_metrics.gmsd_metric(pr, gt, mask=None, reduction="none", data_range=data_range)
        out["deltaE2000"][idx_gt] = ua_metrics.deltaE2000_metric(pr, gt, mask=None, reduction="none", data_range=data_range)

    # Mask-dependent metrics
    if idx_mask_gt is not None and idx_mask_gt.numel() > 0 and batch_gts is not None:
        out["boundary_gmsd_band3"][idx_mask_gt] = ua_metrics.boundary_gmsd(
            batch_preds[idx_mask_gt], batch_gts[idx_mask_gt], mask=batch_masks[idx_mask_gt], band=3, data_range=data_range, reduction="none"
        )
    if idx_mask is not None and idx_mask.numel() > 0:
        out["luminance_suppression_ratio"][idx_mask] = ua_metrics.luminance_suppression_ratio(
            batch_inputs[idx_mask], batch_preds[idx_mask], mask=batch_masks[idx_mask], data_range=data_range, reduction="none"
        )
        out["chroma_consistency_deltaE_ring3"][idx_mask] = ua_metrics.chroma_consistency_deltaE(
            batch_preds[idx_mask], batch_inputs[idx_mask], mask=batch_masks[idx_mask], ring=3, data_range=data_range, reduction="none"
        )

    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark methods using metrics.py")
    parser.add_argument("--config", type=str, default=str(Path(__file__).parents[1] / "configs" / "eval.yaml"))
    args = parser.parse_args()

    cfg = read_yaml(args.config)

    benchmark_dir = Path(cfg["benchmark_dir"]).resolve()
    results_root = benchmark_dir / "results"
    manifest_path = benchmark_dir / "manifests" / "dataset_manifest.parquet"
    per_image_out = results_root / "raw" / "per_image.parquet"
    meta_out = results_root / "meta" / "run_metadata.json"
    ensure_dirs([manifest_path, per_image_out, meta_out])

    device = torch.device(cfg["batch"].get("device", "cuda"))
    torch.set_grad_enabled(False)

    manifest = build_manifest(cfg)
    manifest.to_parquet(manifest_path, index=False)

    rows: List[Dict] = []
    batch_size = int(cfg["batch"].get("size", 8))
    exts = cfg["images"]["extensions"]

    # Iterate method-subdataset pairs for locality
    for (method, subdataset), df_grp in manifest.groupby(["method", "subdataset"], sort=True):
        df_grp = df_grp.reset_index(drop=True)
        indices = df_grp.index.tolist()
        for idx_batch in batch(indices, batch_size):
            df_b = df_grp.loc[idx_batch]

            inputs: List[torch.Tensor] = []
            preds: List[torch.Tensor] = []
            gts: List[torch.Tensor] = []
            masks: List[torch.Tensor] = []
            have_gt_flags: List[bool] = []
            have_mask_flags: List[bool] = []

            paths_input: List[Path] = []
            paths_pred: List[Path] = []
            paths_gt: List[Optional[Path]] = []
            paths_mask: List[Optional[Path]] = []

            for _, r in df_b.iterrows():
                inp_p = Path(r["input_path"])  # exists by construction
                pred_p = Path(r["pred_path"]) if pd.notna(r["pred_path"]) else None
                gt_p = Path(r["gt_path"]) if pd.notna(r["gt_path"]) else None
                mask_p = Path(r["mask_path"]) if pd.notna(r["mask_path"]) else None

                # Skip if pred missing and policy says so
                if pred_p is None or not pred_p.exists():
                    continue

                try:
                    in_t = load_image_tensor(inp_p)
                    pr_t = load_image_tensor(pred_p)
                    if in_t.shape != pr_t.shape:
                        # resize pred to input size if mismatched
                        pr_img = Image.open(pred_p).convert("RGB").resize((in_t.shape[2], in_t.shape[1]), Image.BILINEAR)
                        pr_t = pil_to_torch_chw_uint8(pr_img)
                    inputs.append(in_t)
                    preds.append(pr_t)
                    paths_input.append(inp_p)
                    paths_pred.append(pred_p)

                    if gt_p and gt_p.exists():
                        gt_t = load_image_tensor(gt_p)
                        if gt_t.shape != in_t.shape:
                            gt_img = Image.open(gt_p).convert("RGB").resize((in_t.shape[2], in_t.shape[1]), Image.BILINEAR)
                            gt_t = pil_to_torch_chw_uint8(gt_img)
                        have_gt_flags.append(True)
                        paths_gt.append(gt_p)
                    else:
                        # placeholder (unused in metrics when flag is False)
                        gt_t = in_t
                        have_gt_flags.append(False)
                        paths_gt.append(None)
                    gts.append(gt_t)

                    if mask_p and mask_p.exists():
                        m_img = Image.open(mask_p).convert("L").resize((in_t.shape[2], in_t.shape[1]), Image.NEAREST)
                        m_np = np.array(m_img)
                        m_t = torch.from_numpy((m_np > 0).astype(np.uint8)).unsqueeze(0)  # 1,H,W uint8
                        have_mask_flags.append(True)
                        paths_mask.append(mask_p)
                    else:
                        # placeholder (zeros)
                        m_t = torch.zeros((1, in_t.shape[1], in_t.shape[2]), dtype=torch.uint8)
                        have_mask_flags.append(False)
                        paths_mask.append(None)
                    masks.append(m_t)
                except Exception as e:
                    rows.append(
                        {
                            "subdataset": r["subdataset"],
                            "image_id": r["image_id"],
                            "global_image_key": r["global_image_key"],
                            "method": method,
                            "status": "input_error",
                            "status_reason": str(e),
                        }
                    )

            if len(preds) == 0:
                continue

            B = len(preds)
            batch_inputs = to_bchw_uint8(inputs)
            batch_preds = to_bchw_uint8(preds)
            batch_gts = to_bchw_uint8(gts)
            batch_masks = torch.stack(masks, dim=0).to(dtype=torch.float32)  # B,1,H,W

            # Move to device
            batch_inputs = batch_inputs.to(device)
            batch_preds = batch_preds.to(device)
            batch_gts = batch_gts.to(device)
            batch_masks = batch_masks.to(device)

            start = time.time()
            metrics_out = compute_metrics_for_batch(
                cfg, batch_inputs, batch_preds, batch_gts, batch_masks, have_gt_flags, have_mask_flags, device
            )
            elapsed_ms = (time.time() - start) * 1000.0

            # Materialize per-sample rows
            for bi in range(B):
                base = {
                    "subdataset": df_b.iloc[bi]["subdataset"],
                    "image_id": df_b.iloc[bi]["image_id"],
                    "global_image_key": df_b.iloc[bi]["global_image_key"],
                    "method": method,
                    "has_gt": bool(have_gt_flags[bi]),
                    "has_mask": bool(have_mask_flags[bi]),
                    "device": str(device),
                    "data_range_override": cfg.get("data_range_override"),
                    "compute_time_ms": float(elapsed_ms / B),
                    "status": "ok",
                }
                for k, v in metrics_out.items():
                    base[k] = float(v[bi].detach().to("cpu").item())
                rows.append(base)

    if len(rows) == 0:
        print("No results to write. Check your config and data paths.")
        return

    per_df = pd.DataFrame(rows)
    per_df.sort_values(["subdataset", "image_id", "method"], inplace=True)
    per_df.to_parquet(per_image_out, index=False)

    # Meta
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_dir": str(benchmark_dir),
        "config_path": str(Path(args.config).resolve()),
        "torch_version": torch.__version__,
        "device": str(device),
        "num_rows": int(len(per_df)),
        "num_images": int(per_df[["global_image_key"]].drop_duplicates().shape[0]),
        "num_methods": int(per_df[["method"]].drop_duplicates().shape[0]),
    }
    meta_out.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()


