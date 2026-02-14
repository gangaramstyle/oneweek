"""
Synthetic benchmark for prism data loader methods.

Goal
----
Run many randomized synthetic volumes with varied voxel spacing and shape, then
compare method behavior with metrics split into two groups:

1) Position correctness metrics
   - `relative_patch_centers_pt` agreement (unrotated, patient-space)
   - `relative_patch_centers_pt_rotated` agreement (rotated frame)

2) Patch extraction metrics
   - normalized-patch MAE / percentile error vs naive baseline

It also reports runtime for each method relative to `naive_full_rotate`.

Usage examples
--------------
- Quick smoke run:
  uv run python benchmark_synthetic_loader.py --n-cases 4 --runs-per-case 2

- Larger run with CSV + JSON summary:
  uv run python benchmark_synthetic_loader.py \
    --n-cases 24 \
    --runs-per-case 4 \
    --output-csv benchmark_results/synthetic_runs.csv \
    --summary-json benchmark_results/synthetic_summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from data_loader import nifti_scan


def _make_synthetic_phantom(
    shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a strategically structured synthetic medical-like volume.

    Design choices:
    - smooth low-frequency baseline for interpolation stability
    - multiple ellipsoidal structures with different scales/signs
    - mild edge transitions (tanh) to emulate soft tissue boundaries
    - low-amplitude gaussian noise
    """
    rng = np.random.default_rng(seed)

    x, y, z = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0]),
        np.linspace(-1.0, 1.0, shape[1]),
        np.linspace(-1.0, 1.0, shape[2]),
        indexing="ij",
    )

    volume = (
        60.0 * np.exp(-(x**2 + y**2 + z**2) / 0.55)
        + 20.0 * x
        - 12.0 * y
        + 6.0 * z
    )

    # Add random smooth structures (positive and negative intensities).
    for _ in range(6):
        cx, cy, cz = rng.uniform(-0.65, 0.65, size=3)
        sx, sy, sz = rng.uniform(0.12, 0.28, size=3)
        amplitude = rng.uniform(-30.0, 80.0)
        volume += amplitude * np.exp(
            -(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2 + ((z - cz) / sz) ** 2)
        )

    # Gentle edge-like transitions.
    volume += 15.0 * np.tanh(8.0 * (x + 0.25))
    volume -= 10.0 * np.tanh(7.0 * (z - 0.10))

    # Mild noise.
    volume += rng.normal(0.0, 0.8, size=shape)
    volume = volume.astype(np.float32)

    # Simple diagonal affine with translation.
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing
    affine[:3, 3] = np.array([11.0, -7.0, 25.0], dtype=np.float64)

    return volume, affine


def _choose_center_and_radius(
    scan: nifti_scan,
    candidate_radii_mm: Sequence[float],
) -> Tuple[np.ndarray, float]:
    """Find a valid center/radius pair that satisfies loader constraints."""
    for radius in candidate_radii_mm:
        try:
            center = scan.get_random_center_idx(float(radius), scan.patch_shape)
            return np.asarray(center, dtype=np.int64), float(radius)
        except ValueError:
            continue
    raise RuntimeError("Could not find valid center for any candidate radius.")


def _summarize_method(rows: pd.DataFrame, method: str) -> Dict[str, float]:
    """Compute aggregate metrics for one comparison method."""
    subset = rows[rows["method"] == method]
    if subset.empty:
        return {
            "method": method,
            "n": 0,
        }

    def _q(col: str, q: float) -> float:
        return float(np.percentile(subset[col].to_numpy(dtype=np.float64), q))

    return {
        "method": method,
        "n": int(len(subset)),
        "patch_norm_mae_mean": float(subset["patch_norm_mae"].mean()),
        "patch_norm_mae_p95": _q("patch_norm_mae", 95),
        "patch_norm_abs_p95_mean": float(subset["patch_norm_abs_p95"].mean()),
        "patch_norm_abs_p95_p95": _q("patch_norm_abs_p95", 95),
        "position_unrot_max_mm": float(subset["position_unrot_max_err_mm"].max()),
        "position_rot_max_mm": float(subset["position_rot_max_err_mm"].max()),
        "time_method_median_s": float(np.median(subset["time_method_s"].to_numpy(dtype=np.float64))),
        "time_naive_median_s": float(np.median(subset["time_naive_s"].to_numpy(dtype=np.float64))),
        "speed_ratio_method_over_naive_median": float(
            np.median(subset["speed_ratio_method_over_naive"].to_numpy(dtype=np.float64))
        ),
        "speedup_naive_over_method_median": float(
            np.median(subset["speedup_naive_over_method"].to_numpy(dtype=np.float64))
        ),
    }


def run_benchmark(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rng = np.random.default_rng(args.seed)

    # Candidate shapes/spacings intentionally include anisotropic cases.
    shapes: List[Tuple[int, int, int]] = [
        (96, 96, 64),
        (88, 72, 48),
        (80, 80, 40),
        (72, 64, 56),
    ]
    spacings: List[Tuple[float, float, float]] = [
        (0.60, 0.60, 1.00),
        (0.75, 0.75, 2.50),
        (1.00, 1.00, 5.00),
        (1.20, 0.90, 2.00),
        (0.90, 0.90, 1.50),
    ]

    # Rotation bounds applied per run as random uniform draw.
    rot_bounds = np.asarray([args.rot_bound_x, args.rot_bound_y, args.rot_bound_z], dtype=np.float64)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {"optimized_fused", "optimized_local"}
    for m in methods:
        if m not in allowed:
            raise ValueError(f"Unsupported compare method '{m}'. Allowed: {sorted(allowed)}")

    run_rows: List[Dict[str, object]] = []
    skipped_runs = 0

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for case_idx in range(args.n_cases):
            shape = shapes[case_idx % len(shapes)]
            spacing = spacings[(case_idx * 3) % len(spacings)]
            phantom_seed = int(rng.integers(0, 2**31 - 1))

            volume, affine = _make_synthetic_phantom(
                shape=shape,
                spacing=spacing,
                seed=phantom_seed,
            )

            scan_path = tmpdir_path / f"synthetic_case_{case_idx:04d}.nii.gz"
            nib.save(nib.Nifti1Image(volume, affine), str(scan_path))

            med = float(np.median(volume))
            std = float(np.std(volume))

            scan = nifti_scan(
                path_to_scan=str(scan_path),
                median=med,
                stdev=std,
                base_patch_size=args.base_patch_size_mm,
                modality=args.modality,
            )

            for run_in_case in range(args.runs_per_case):
                run_seed = int(rng.integers(0, 2**31 - 1))

                # Draw random rotation within configured bounds.
                rot = tuple(
                    rng.uniform(low=-rot_bounds, high=rot_bounds).astype(np.float64).tolist()
                )

                try:
                    center_vox, radius_mm = _choose_center_and_radius(
                        scan,
                        candidate_radii_mm=[float(v) for v in args.radius_candidates_mm],
                    )
                except RuntimeError:
                    skipped_runs += 1
                    continue

                common = dict(
                    n_patches=int(args.n_patches),
                    subset_center=tuple(int(v) for v in center_vox.tolist()),
                    sampling_radius_mm=float(radius_mm),
                    rotation_degrees=rot,
                    seed=run_seed,
                    wc=float(args.window_center),
                    ww=float(args.window_width),
                )

                # Baseline timing + outputs.
                t0 = time.perf_counter()
                naive = scan.train_sample(method="naive_full_rotate", **common)
                t1 = time.perf_counter()
                time_naive = max(t1 - t0, 1e-12)

                patch_centers = naive["patch_centers_vox"]

                for method in methods:
                    t2 = time.perf_counter()
                    out = scan.train_sample(
                        method=method,
                        patch_centers_vox=patch_centers,
                        **common,
                    )
                    t3 = time.perf_counter()
                    time_method = max(t3 - t2, 1e-12)

                    # Position metrics (separate objective family).
                    pos_unrot_max = float(
                        np.max(
                            np.abs(
                                naive["relative_patch_centers_pt"]
                                - out["relative_patch_centers_pt"]
                            )
                        )
                    )
                    pos_rot_max = float(
                        np.max(
                            np.abs(
                                naive["relative_patch_centers_pt_rotated"]
                                - out["relative_patch_centers_pt_rotated"]
                            )
                        )
                    )

                    # Patch metrics in normalized space (independent objective family).
                    abs_diff = np.abs(
                        np.asarray(naive["normalized_patches_3d"], dtype=np.float32)
                        - np.asarray(out["normalized_patches_3d"], dtype=np.float32)
                    )
                    patch_norm_mae = float(np.mean(abs_diff))
                    patch_norm_abs_p95 = float(np.percentile(abs_diff, 95))

                    run_rows.append(
                        {
                            "case_idx": int(case_idx),
                            "run_in_case": int(run_in_case),
                            "method": method,
                            "shape_x": int(shape[0]),
                            "shape_y": int(shape[1]),
                            "shape_z": int(shape[2]),
                            "spacing_x_mm": float(spacing[0]),
                            "spacing_y_mm": float(spacing[1]),
                            "spacing_z_mm": float(spacing[2]),
                            "patch_shape_mm": str(tuple(float(v) for v in scan.patch_shape)),
                            "n_patches": int(args.n_patches),
                            "sampling_radius_mm": float(radius_mm),
                            "rotation_x_deg": float(rot[0]),
                            "rotation_y_deg": float(rot[1]),
                            "rotation_z_deg": float(rot[2]),
                            "seed": int(run_seed),
                            "time_naive_s": float(time_naive),
                            "time_method_s": float(time_method),
                            "speed_ratio_method_over_naive": float(time_method / time_naive),
                            "speedup_naive_over_method": float(time_naive / time_method),
                            "position_unrot_max_err_mm": pos_unrot_max,
                            "position_rot_max_err_mm": pos_rot_max,
                            "patch_norm_mae": patch_norm_mae,
                            "patch_norm_abs_p95": patch_norm_abs_p95,
                        }
                    )

    df = pd.DataFrame(run_rows)

    summary = {
        "config": {
            "seed": int(args.seed),
            "n_cases": int(args.n_cases),
            "runs_per_case": int(args.runs_per_case),
            "n_patches": int(args.n_patches),
            "base_patch_size_mm": int(args.base_patch_size_mm),
            "methods": methods,
            "modality": str(args.modality),
            "window_center": float(args.window_center),
            "window_width": float(args.window_width),
            "radius_candidates_mm": [float(v) for v in args.radius_candidates_mm],
            "rotation_bounds_deg": {
                "x": float(args.rot_bound_x),
                "y": float(args.rot_bound_y),
                "z": float(args.rot_bound_z),
            },
        },
        "total_rows": int(len(df)),
        "skipped_runs": int(skipped_runs),
        "methods": {
            m: _summarize_method(df, m)
            for m in methods
        },
    }

    return df, summary


def _print_summary(summary: Dict[str, object]) -> None:
    print("\n=== Synthetic Loader Benchmark Summary ===")
    print(f"rows: {summary['total_rows']} | skipped_runs: {summary['skipped_runs']}")

    methods = summary.get("methods", {})
    for method, stats in methods.items():
        if stats.get("n", 0) == 0:
            print(f"\n[{method}] no data")
            continue
        print(f"\n[{method}] n={stats['n']}")
        print(
            "patch_norm_mae mean={:.6f} p95={:.6f}".format(
                stats["patch_norm_mae_mean"],
                stats["patch_norm_mae_p95"],
            )
        )
        print(
            "patch_norm_abs_p95 mean={:.6f} p95={:.6f}".format(
                stats["patch_norm_abs_p95_mean"],
                stats["patch_norm_abs_p95_p95"],
            )
        )
        print(
            "position max err (unrot/rot) = {:.6e} / {:.6e} mm".format(
                stats["position_unrot_max_mm"],
                stats["position_rot_max_mm"],
            )
        )
        print(
            "time median naive={:.6f}s method={:.6f}s ratio(method/naive)={:.4f} speedup(naive/method)={:.2f}x".format(
                stats["time_naive_median_s"],
                stats["time_method_median_s"],
                stats["speed_ratio_method_over_naive_median"],
                stats["speedup_naive_over_method_median"],
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic benchmark for prism loader methods.")

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-cases", type=int, default=12, help="Number of synthetic volumes.")
    parser.add_argument("--runs-per-case", type=int, default=3, help="Random runs per volume.")
    parser.add_argument("--n-patches", type=int, default=32)
    parser.add_argument("--base-patch-size-mm", type=int, default=16)

    parser.add_argument(
        "--methods",
        type=str,
        default="optimized_fused,optimized_local",
        help="Comma-separated methods to compare against naive baseline.",
    )

    parser.add_argument("--modality", type=str, default="CT", choices=["CT", "MR"])

    parser.add_argument("--window-center", type=float, default=0.0)
    parser.add_argument("--window-width", type=float, default=250.0)

    parser.add_argument("--rot-bound-x", type=float, default=20.0)
    parser.add_argument("--rot-bound-y", type=float, default=20.0)
    parser.add_argument("--rot-bound-z", type=float, default=20.0)

    parser.add_argument(
        "--radius-candidates-mm",
        type=float,
        nargs="+",
        default=[22.0, 18.0, 14.0, 10.0, 8.0],
        help="Tried in order until a valid center is found.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark_results/synthetic_loader_runs.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional path for aggregate JSON summary.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    df, summary = run_benchmark(args)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _print_summary(summary)
    print(f"\nPer-run CSV written: {output_csv}")
    if args.summary_json:
        print(f"Summary JSON written: {args.summary_json}")


if __name__ == "__main__":
    main()
