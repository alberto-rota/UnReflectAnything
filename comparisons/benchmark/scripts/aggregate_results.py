import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def read_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def aggregate(per_image_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(per_image_path)

    # Metrics columns (exclude meta)
    exclude = {"subdataset", "image_id", "global_image_key", "method", "has_gt", "has_mask", "device", "data_range_override", "compute_time_ms", "status", "status_reason"}
    metric_cols = [c for c in df.columns if c not in exclude]

    # Per method per subdataset
    grp_cols = ["subdataset", "method"]
    stats = df.groupby(grp_cols, dropna=False)[metric_cols].agg([
        ("mean", "mean"),
        ("std", "std"),
        ("median", "median"),
        ("iqr", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        ("num_images", "count"),
    ])
    stats.columns = [f"{m}_{s}" for m, s in stats.columns]
    stats = stats.reset_index()
    (out_dir / "per_method_subdataset.parquet").parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(out_dir / "per_method_subdataset.parquet", index=False)

    # Per method overall
    grp_cols_overall = ["method"]
    stats_overall = df.groupby(grp_cols_overall, dropna=False)[metric_cols].agg([
        ("mean", "mean"),
        ("std", "std"),
        ("median", "median"),
        ("iqr", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        ("num_images", "count"),
    ])
    stats_overall.columns = [f"{m}_{s}" for m, s in stats_overall.columns]
    stats_overall = stats_overall.reset_index()
    stats_overall.to_parquet(out_dir / "per_method_overall.parquet", index=False)

    # Rankings (lower-is-better or higher?)
    lower_better = {"mse", "gmsd", "deltaE2000", "brisque", "niqe", "piqe", "boundary_gmsd_band3"}
    higher_better = {"psnr", "ssim", "luminance_suppression_ratio"}
    # lpips and dists are lower-better
    lower_better |= {"lpips_vgg", "dists"}

    rankings_rows = []
    # Per subdataset
    for sub, sdf in df.groupby("subdataset"):
        for metric in metric_cols:
            if metric not in lower_better and metric not in higher_better:
                continue
            agg = sdf.groupby("method")[metric].mean().dropna()
            if metric in lower_better:
                order = agg.sort_values(ascending=True)
            else:
                order = agg.sort_values(ascending=False)
            for rank, (method, val) in enumerate(order.items(), start=1):
                rankings_rows.append({"scope": sub, "metric": metric, "method": method, "rank": rank, "mean_value": float(val)})

    # Overall
    for metric in metric_cols:
        if metric not in lower_better and metric not in higher_better:
            continue
        agg = df.groupby("method")[metric].mean().dropna()
        order = agg.sort_values(ascending=(metric in lower_better))
        for rank, (method, val) in enumerate(order.items(), start=1):
            rankings_rows.append({"scope": "overall", "metric": metric, "method": method, "rank": rank, "mean_value": float(val)})

    rankings = pd.DataFrame(rankings_rows)
    (out_dir.parent / "rankings").mkdir(parents=True, exist_ok=True)
    rankings.to_parquet(out_dir.parent / "rankings" / "rank_tables.parquet", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_image", type=str, required=False, help="Path to per_image.parquet")
    parser.add_argument("--benchmark_dir", type=str, required=False, help="Path to benchmark root")
    args = parser.parse_args()

    if args.per_image:
        per_image_path = Path(args.per_image)
        benchmark_dir = per_image_path.parents[2]
    else:
        benchmark_dir = Path(args.benchmark_dir)
        per_image_path = benchmark_dir / "results" / "raw" / "per_image.parquet"

    out_dir = benchmark_dir / "results" / "aggregates"
    aggregate(per_image_path, out_dir)


if __name__ == "__main__":
    main()


