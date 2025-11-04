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
import re
import itertools
from scipy import stats

# Ensure project root on sys.path to import metrics.py reliably.
# Robustly walk up until a folder containing metrics.py is found.
def _find_repo_root_with_metrics(start: Path, max_hops: int = 8) -> Path:
    cur = start.resolve()
    for _ in range(max_hops):
        if (cur / "metrics.py").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback to historical assumption: 3 levels up from this script
    return start.resolve().parents[3]

ROOT = _find_repo_root_with_metrics(Path(__file__).parent)
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


def list_images_in_subdataset(folder: Path, exts: List[str], recursive: bool = False) -> List[Path]:
    if not folder.exists():
        return []
    if recursive:
        # Collect all files under folder matching extensions
        return sorted([p for p in folder.rglob("*") if p.is_file() and is_image_file(p, exts)])
    return sorted([p for p in folder.iterdir() if p.is_file() and is_image_file(p, exts)])


def _natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


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


def locate_mask_for(
    stem: str,
    subdataset: str,
    cfg_masks: dict,
    benchmark_dir: Path,
    relative_path_under_sub: Optional[Path] = None,
) -> Optional[Path]:
    if not cfg_masks.get("enabled", False):
        return None
    base_dir = cfg_masks.get("base_dir")
    glob_pattern = cfg_masks.get("glob_pattern")
    if base_dir:
        if relative_path_under_sub is not None:
            # Mirror the input tree; default mask extension to .png
            candidate = Path(base_dir) / subdataset / relative_path_under_sub.with_suffix(".png")
        else:
            candidate = Path(base_dir) / subdataset / f"{stem}.png"
        return candidate if candidate.exists() else None
    if glob_pattern:
        rel_str = str(relative_path_under_sub) if relative_path_under_sub is not None else ""
        pat = glob_pattern.format(
            benchmark_dir=str(benchmark_dir), subdataset=subdataset, stem=stem, relative_path=rel_str
        )
        matches = list(Path(benchmark_dir).glob(pat))
        return matches[0] if matches else None
    return None


def ensure_dirs(paths: List[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


def _resolve_results_root(cfg: dict, benchmark_dir: Path) -> Path:
    # Prefer cfg["results_dir"], fallback to cfg["output_dir"], else benchmark_dir/results
    cfg_out = cfg.get("results_dir") or cfg.get("output_dir")
    if cfg_out is None or str(cfg_out).strip() == "":
        return benchmark_dir / "results"
    p = Path(cfg_out)
    if not p.is_absolute():
        p = (benchmark_dir / p).resolve()
    return p


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    adj = pvals * m / ranks
    for i in range(m - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return np.clip(adj, 0, 1)


def _cliff_delta(a: np.ndarray, b: np.ndarray) -> float:
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    i = j = more = less = 0
    na, nb = len(a_sorted), len(b_sorted)
    while i < na and j < nb:
        if a_sorted[i] > b_sorted[j]:
            more += na - i
            j += 1
        elif a_sorted[i] < b_sorted[j]:
            less += nb - j
            i += 1
        else:
            i += 1
            j += 1
    return (more - less) / (na * nb)


def _run_stat_tests(per_df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    exclude = {"subdataset", "image_id", "global_image_key", "method", "has_gt", "has_mask", "device", "data_range_override", "compute_time_ms", "status", "status_reason"}
    metric_cols = [c for c in per_df.columns if c not in exclude]
    frames = []
    scopes = [(name, g) for name, g in per_df.groupby("subdataset")]
    scopes.append(("overall", per_df))
    for scope_name, g in scopes:
        methods = sorted(g["method"].unique())
        for metric in metric_cols:
            pivot = g.pivot_table(index="global_image_key", columns="method", values=metric, aggfunc="mean")
            if pivot.empty:
                continue
            present_methods = [m for m in methods if m in pivot.columns]
            if len(present_methods) < 2:
                continue
            for a, b in itertools.combinations(present_methods, 2):
                ab = pivot[[a, b]].dropna()
                n = len(ab)
                if n < 3:
                    continue
                x = ab[a].to_numpy()
                y = ab[b].to_numpy()
                try:
                    w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                    t = stats.ttest_rel(x, y, nan_policy="omit")
                    effect = _cliff_delta(x, y)
                    frames.append({
                        "scope": scope_name,
                        "metric": metric,
                        "method_a": a,
                        "method_b": b,
                        "n": int(n),
                        "p_wilcoxon": float(w.pvalue),
                        "p_ttest": float(t.pvalue),
                        "effect_size": float(effect),
                    })
                except Exception:
                    continue
    out = pd.DataFrame(frames)
    if not out.empty:
        out["p_min"] = out[["p_wilcoxon", "p_ttest"]].min(axis=1)
        out["p_adj"] = _benjamini_hochberg(out["p_min"].to_numpy())
        out["significant"] = out["p_adj"] < 0.05
    stats_dir = out_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(stats_dir / "pairwise_tests.parquet", index=False)
    return out


def _write_summary(per_df: pd.DataFrame, stats_df: pd.DataFrame, out_root: Path) -> None:
    exclude = {"subdataset", "image_id", "global_image_key", "method", "has_gt", "has_mask", "device", "data_range_override", "compute_time_ms", "status", "status_reason"}
    metric_cols = [c for c in per_df.columns if c not in exclude]
    greater_is_better = {
        "psnr": True, "ssim": True,
        "mse": False, "dists": False, "gmsd": False, "deltaE2000": False,
        "brisque": False, "niqe": False, "piqe": False,
        "boundary_gmsd_band3": False, "luminance_suppression_ratio": False, "chroma_consistency_deltaE_ring3": False,
    }
    # Overall means per method
    overall = per_df.groupby("method")[metric_cols].mean(numeric_only=True)
    # Determine best per metric
    best_map = {}
    for m in metric_cols:
        if m not in overall.columns:
            continue
        if greater_is_better.get(m, False):
            best = overall[m].idxmax()
        else:
            best = overall[m].idxmin()
        best_map[m] = best

    lines = []
    lines.append(f"# Evaluation Summary\n")
    lines.append(f"Methods: {', '.join(sorted(per_df['method'].unique()))}\n")
    lines.append(f"Num images: {per_df[['global_image_key']].drop_duplicates().shape[0]}\n")
    lines.append(f"Num rows: {len(per_df)}\n")
    lines.append("\n## Overall means by method\n")
    lines.append("| Metric | " + " | ".join(overall.index.tolist()) + " | Best |\n")
    lines.append("|---|" + "|".join(["---"] * (len(overall.index) + 1)) + "|\n")
    for m in metric_cols:
        if m not in overall.columns:
            continue
        vals = [f"{overall.loc[method, m]:.4f}" if m in overall.columns else "-" for method in overall.index]
        best_method = best_map.get(m, "-")
        lines.append("| " + m + " | " + " | ".join(vals) + f" | {best_method} |\n")

    # Per-subdataset means
    lines.append("\n## Per-subdataset means (by method)\n")
    for sub, g in per_df.groupby("subdataset"):
        sub_means = g.groupby("method")[metric_cols].mean(numeric_only=True)
        lines.append(f"\n### {sub}\n")
        lines.append("| Metric | " + " | ".join(sub_means.index.tolist()) + " |\n")
        lines.append("|---|" + "|".join(["---"] * (len(sub_means.index))) + "|\n")
        for m in metric_cols:
            if m not in sub_means.columns:
                continue
            vals = [f"{sub_means.loc[method, m]:.4f}" if m in sub_means.columns else "-" for method in sub_means.index]
            lines.append("| " + m + " | " + " | ".join(vals) + " |\n")

    # Statistical tests summary
    lines.append("\n## Statistical tests (Wilcoxon/paired t-test with BH correction)\n")
    if stats_df is not None and not stats_df.empty:
        sig = stats_df[stats_df["significant"]]
        lines.append(f"Significant pairs (adj p < 0.05): {len(sig)} / {len(stats_df)}\n")
        head = sig.sort_values(["p_adj"]).head(50)
        for _, r in head.iterrows():
            lines.append(
                f"- [{r['scope']}] {r['metric']}: {r['method_a']} vs {r['method_b']} (n={int(r['n'])}), p_adj={r['p_adj']:.3e}, effect={r['effect_size']:.3f}\n"
            )
    else:
        lines.append("No significant differences detected or insufficient data.\n")

    report_dir = out_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.md").write_text("".join(lines))


def build_manifest(cfg: dict) -> pd.DataFrame:
    """Build the evaluation manifest from the benchmark directory.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration. Expected keys: `benchmark_dir`, `methods`,
        `images`, optional `masks`, and `policy`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per (method, subdataset, image). Columns:
        - `subdataset`
        - `image_id`
        - `global_image_key`
        - `input_path`
        - `gt_path`
        - `mask_path`
        - `method`
        - `pred_path`
        - `has_input`
        - `has_gt`
        - `has_mask`
        - `has_pred`

        The DataFrame may be empty, but these columns will always exist to
        allow downstream sorting and grouping without KeyError.
    """
    benchmark_dir = Path(cfg["benchmark_dir"]).resolve()
    input_root = benchmark_dir / "input"
    gt_root = benchmark_dir / "diffuse_gt"

    methods_cfg = [MethodConfig(**m) for m in cfg["methods"] if m.get("enabled", True)]
    print(f"Methods to evaluate: {[m.name for m in methods_cfg]}")
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Input root: {input_root}")
    print(f"GT root: {gt_root}")
    print(f"Extensions: {cfg['images']['extensions']}")
    print(f"Subdatasets: {list_subdatasets(input_root)}")
    exts = cfg["images"]["extensions"]
    recursive = bool(cfg["images"].get("recursive", False))

    subdatasets = list_subdatasets(input_root)

    rows: List[Dict] = []
    for sub in subdatasets:
        input_sub = input_root / sub
        gt_sub = gt_root / sub
        input_images = list_images_in_subdataset(input_sub, exts, recursive=recursive)
        # Auto-fallback: if nothing found with non-recursive, try recursive search
        if not input_images and not recursive:
            input_images = list_images_in_subdataset(input_sub, exts, recursive=True)
        # Natural sort by relative path (for stability across OS)
        input_images = sorted(
            input_images,
            key=lambda p: _natural_key(str(p.relative_to(input_sub)) if p.is_relative_to(input_sub) else str(p)),
        )

        # Prepare GT/mask paths per input image
        per_input_meta: List[Tuple[Path, Optional[Path], Optional[Path], str, Optional[Path]]] = []
        for inp_path in input_images:
            stem = inp_path.stem
            try:
                rel_under_sub = inp_path.relative_to(input_sub)
            except Exception:
                rel_under_sub = None

            if gt_sub.exists():
                if rel_under_sub is not None:
                    gt_path = gt_sub / rel_under_sub
                else:
                    gt_path = gt_sub / f"{stem}{inp_path.suffix}"
            else:
                gt_path = None
            gt_path = gt_path if (gt_path and gt_path.exists()) else None
            mask_path = locate_mask_for(stem, sub, cfg.get("masks", {}), benchmark_dir, rel_under_sub)
            per_input_meta.append((inp_path, gt_path, mask_path, stem, rel_under_sub))

        # For each method, collect its predictions and pair by natural-sorted order
        for m in methods_cfg:
            pred_root = benchmark_dir / m.output_dir_name / sub
            pred_images = list_images_in_subdataset(pred_root, exts, recursive=recursive)
            if not pred_images and not recursive:
                pred_images = list_images_in_subdataset(pred_root, exts, recursive=True)
            pred_images = sorted(
                pred_images,
                key=lambda p: _natural_key(str(p.relative_to(pred_root)) if p.is_relative_to(pred_root) else str(p)),
            )

            pair_count = min(len(per_input_meta), len(pred_images))
            for i in range(pair_count):
                inp_path, gt_path, mask_path, stem, rel_under_sub = per_input_meta[i]
                pred_path = pred_images[i]
                rows.append(
                    {
                        "subdataset": sub,
                        "image_id": stem,
                        "global_image_key": f"{sub}/{stem}",
                        "input_path": str(inp_path),
                        "gt_path": str(gt_path) if gt_path else None,
                        "mask_path": str(mask_path) if mask_path else None,
                        "method": m.name,
                        "pred_path": str(pred_path),
                        "has_input": True,
                        "has_gt": gt_path is not None,
                        "has_mask": mask_path is not None,
                        "has_pred": True,
                    }
                )
    # Ensure required columns exist even when there are no rows
    expected_columns = [
        "subdataset",
        "image_id",
        "global_image_key",
        "input_path",
        "gt_path",
        "mask_path",
        "method",
        "pred_path",
        "has_input",
        "has_gt",
        "has_mask",
        "has_pred",
    ]
    if len(rows) == 0:
        return pd.DataFrame(columns=expected_columns)

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
    metrics_cfg = cfg.get("metrics", {})
    fr_list = metrics_cfg.get("full_reference", []) or []
    compute_lpips = any(m.lower() == "lpips_vgg" for m in fr_list)
    lpips_net = str(metrics_cfg.get("lpips_net", "alex"))

    B = batch_preds.shape[0]
    # Initialize all outputs as NaN (on device for assignment, move to cpu later)
    def nan_vec():
        return torch.full((B,), float("nan"), device=device, dtype=torch.float32)

    out: Dict[str, torch.Tensor] = {
        "mse": nan_vec(), "psnr": nan_vec(), "ssim": nan_vec(),
        "dists": nan_vec(), "gmsd": nan_vec(), "deltaE2000": nan_vec(),
        "brisque": nan_vec(), "niqe": nan_vec(), "piqe": nan_vec(),
        "boundary_gmsd_band3": nan_vec(), "luminance_suppression_ratio": nan_vec(),
        "chroma_consistency_deltaE_ring3": nan_vec(),
    }
    if compute_lpips:
        out["lpips_vgg"] = nan_vec()

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
        # if compute_lpips:
        #     # LPIPS backbone selectable via config: metrics.lpips_net (default 'alex')
        #     out["lpips_vgg"][idx_gt] = ua_metrics.lpips_metric(pr, gt, mask=None, net=lpips_net, reduction="none", data_range=data_range)
        # print("DISTS")
        # out["dists"][idx_gt] = ua_metrics.dists_metric(pr, gt, mask=None, reduction="none", data_range=data_range)
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
    results_root = _resolve_results_root(cfg, benchmark_dir)
    manifest_path = results_root / "manifests" / "dataset_manifest.parquet"
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
    print(f"Manifest: {manifest}")
    # Iterate method-subdataset pairs for locality
    for (method, subdataset), df_grp in manifest.groupby(["method", "subdataset"], sort=True):
        print(f"Evaluating {method} on {subdataset}")
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

    # Run statistical tests and write summary report
    stats_df = _run_stat_tests(per_df, results_root)
    _write_summary(per_df, stats_df, results_root)

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


