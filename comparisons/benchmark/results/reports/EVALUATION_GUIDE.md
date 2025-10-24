## Evaluation Guide

This guide explains how to run the benchmark evaluation using `metrics.py`, which metrics are computed, and what files are produced.

### Prerequisites

- Dependencies in `requirements.txt` installed. Notable extras used by metrics: `kornia`, `piq`, `lpips`, `scikit-image`, `pyarrow`, `pandas`, `scipy`.
- Benchmark directory with the following structure under the path configured in `configs/eval.yaml`:

```
benchmark/
  input/
    <subdataset>/
      <image>.(png|jpg|...)
  diffuse_gt/                # optional, may be missing per image or entirely
    <subdataset>/
      <image>.(png|jpg|...)
  MethodA_output/
    <subdataset>/
      <image>.(png|jpg|...)
  MethodB_output/
    <subdataset>/
      <image>.(png|jpg|...)
  ...
```

If masks are available, configure their location (see below). Otherwise mask-dependent metrics are skipped.

### Configuration: `comparisons/benchmark/configs/eval.yaml`

Key fields:

- `benchmark_dir`: absolute path to `benchmark/` root containing `input/`, optional `diffuse_gt/`, and each `*_output/` method folder.
- `methods`: list of methods to evaluate, with `name` and `output_dir_name` (folder name). Set `enabled: true` to include.
- `images.extensions`: set of image extensions to scan.
- `masks`: optional. Provide `base_dir` or a `glob_pattern` template; if disabled or not found per image, mask-dependent metrics are null.
- `batch`: `size` and `device` (e.g., `cuda`, `cuda:0`, or `cpu`).
- `metrics`: which metric families to compute. Full-reference metrics require ground truth per image.
- `data_range_override`: null means auto inference; set to `1.0` or `255.0` to override.
- `policy`: matching and skipping behavior when files are missing.

### Run evaluation

From the project root:

```bash
python comparisons/benchmark/scripts/evaluate_benchmark.py --config comparisons/benchmark/configs/eval.yaml
```

This will:

1. Build a manifest of evaluable items keyed by `(subdataset, image_id, method)` and write it to:
   - `benchmark/manifests/dataset_manifest.parquet`
2. Compute metrics in GPU batches and write per-image results to:
   - `benchmark/results/raw/per_image.parquet`
3. Save run metadata to:
   - `benchmark/results/meta/run_metadata.json`

Then aggregate and produce rankings:

```bash
python comparisons/benchmark/scripts/aggregate_results.py --benchmark_dir /path/to/benchmark
```

And compute pairwise statistical tests:

```bash
python comparisons/benchmark/scripts/stat_tests.py --benchmark_dir /path/to/benchmark
```

### Metrics computed

- Full-reference (require `diffuse_gt`): `mse`, `psnr`, `ssim`, `lpips_vgg`, `dists`, `gmsd`, `deltaE2000`.
- No-reference: `brisque`, `niqe`, `piqe`.
- Mask-dependent: `boundary_gmsd_band3` (needs GT), `luminance_suppression_ratio`, `chroma_consistency_deltaE_ring3`.

Unavailable inputs (e.g., missing GT or mask) yield nulls for the corresponding metrics. Method predictions missing are skipped (policy configurable).

### Outputs and their content

- `benchmark/manifests/dataset_manifest.parquet`
  - One row per `(subdataset, image_id, method)`. Columns:
    - `subdataset`, `image_id`, `global_image_key = "subdataset/image_id"`
    - `input_path`, `gt_path` (nullable), `mask_path` (nullable)
    - `method`, `pred_path` (nullable)
    - `has_input`, `has_gt`, `has_mask`, `has_pred`

- `benchmark/results/raw/per_image.parquet`
  - One row per `(subdataset, image_id, method)`. Columns:
    - Keys: `subdataset`, `image_id`, `global_image_key`, `method`
    - Flags: `has_gt`, `has_mask`
    - Meta: `device`, `data_range_override`, `compute_time_ms`, `status`, `status_reason` (nullable)
    - Metrics: numeric columns for each enabled metric (null where not applicable)

- `benchmark/results/aggregates/per_method_subdataset.parquet`
  - Aggregated stats per `(subdataset, method)`: for each metric, `mean`, `std`, `median`, `iqr`, `num_images`.

- `benchmark/results/aggregates/per_method_overall.parquet`
  - Aggregated stats per `method` across all subdatasets with same fields as above.

- `benchmark/results/rankings/rank_tables.parquet`
  - Per-metric rankings within each subdataset and overall. Columns: `scope`, `metric`, `method`, `rank`, `mean_value`.

- `benchmark/results/stats/pairwise_tests.parquet`
  - Paired tests between methods per metric and scope (`subdataset` and `overall`). Columns:
    - `scope`, `metric`, `method_a`, `method_b`, `n`, `p_wilcoxon`, `p_ttest`, `effect_size`, `p_adj`, `significant`.

- `benchmark/results/meta/run_metadata.json`
  - Run metadata: timestamp, paths, torch version, device, sizes.

### Notes

- Subdataset alignment uses `benchmark/input/<subdataset>/` as the authoritative list. Predictions are matched by basename within the same subdataset. Ground truth is optional and used when present.
- All tensors follow BCHW convention and are processed on GPU where possible.
- Parquet is used for data tables; you can load them with Pandas/Polars/DuckDB for further analysis and statistical testing.


