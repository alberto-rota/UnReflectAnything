import argparse
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy import stats


def cliff_delta(a: np.ndarray, b: np.ndarray) -> float:
    # Efficient Cliff's delta for moderate sizes
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


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    adj = pvals * m / ranks
    # monotonic
    for i in range(m - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return np.clip(adj, 0, 1)


def pairwise_tests(df: pd.DataFrame, scope_name: str) -> pd.DataFrame:
    exclude = {"subdataset", "image_id", "global_image_key", "method", "has_gt", "has_mask", "device", "data_range_override", "compute_time_ms", "status", "status_reason"}
    metric_cols = [c for c in df.columns if c not in exclude]
    rows = []
    methods = sorted(df["method"].unique())

    for metric in metric_cols:
        # Build per-image wide table for matched pairs
        pivot = df.pivot_table(index="global_image_key", columns="method", values=metric, aggfunc="mean")
        # Drop rows with any NaN among methods in pairs during each pair test
        for a, b in itertools.combinations(methods, 2):
            ab = pivot[[a, b]].dropna()
            n = len(ab)
            if n < 3:
                continue
            x = ab[a].to_numpy()
            y = ab[b].to_numpy()
            try:
                w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                t = stats.ttest_rel(x, y, nan_policy="omit")
                effect = cliff_delta(x, y)
                rows.append({
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

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_min"] = out[["p_wilcoxon", "p_ttest"]].min(axis=1)
        out["p_adj"] = benjamini_hochberg(out["p_min"].to_numpy())
        out["significant"] = out["p_adj"] < 0.05
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_image", type=str, required=False)
    parser.add_argument("--benchmark_dir", type=str, required=False)
    args = parser.parse_args()

    if args.per_image:
        per_image_path = Path(args.per_image)
        benchmark_dir = per_image_path.parents[2]
    else:
        benchmark_dir = Path(args.benchmark_dir)
        per_image_path = benchmark_dir / "results" / "raw" / "per_image.parquet"

    df = pd.read_parquet(per_image_path)

    rows = []
    # Per subdataset scopes
    all_scopes = [
        (name, g) for name, g in df.groupby("subdataset")
    ]
    all_scopes.append(("overall", df))

    out_dir = benchmark_dir / "results" / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name, g in all_scopes:
        frames.append(pairwise_tests(g, name))
    all_pairs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_pairs.to_parquet(out_dir / "pairwise_tests.parquet", index=False)


if __name__ == "__main__":
    main()


