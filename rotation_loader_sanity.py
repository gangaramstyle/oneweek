import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")

with app.setup:
    import time
    import tracemalloc
    from typing import Tuple

    import marimo as mo
    import nibabel as nib
    import numpy as np

    from data_loader import nifti_scan


@app.cell
def _():
    def parse_triplet(text: str, cast=float) -> Tuple[float, float, float]:
        values = [cast(v.strip()) for v in text.split(",")]
        if len(values) != 3:
            raise ValueError(f"Expected 3 comma-separated values, got: {text}")
        return values[0], values[1], values[2]

    def normalize_to_01(values: np.ndarray) -> np.ndarray:
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        if v_max <= v_min:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - v_min) / (v_max - v_min)).astype(np.float32)

    return normalize_to_01, parse_triplet


@app.cell(hide_code=True)
def _():
    path_input = mo.ui.text(
        value="",
        label="Path to NIfTI (.nii/.nii.gz)",
        full_width=True,
    )
    modality_input = mo.ui.dropdown(
        options=["CT", "MR"],
        value="CT",
        label="Modality",
    )
    seed_input = mo.ui.number(start=0, stop=10_000_000, value=1234, label="Seed")

    center_input = mo.ui.text(
        value="",
        label="Center voxel i,j,k (blank=random)",
    )

    base_patch_size_input = mo.ui.number(
        start=1,
        stop=256,
        value=16,
        step=1,
        label="Base patch size (mm)",
    )

    n_patches_input = mo.ui.number(
        start=1,
        stop=512,
        value=64,
        step=1,
        label="N patches",
    )

    radius_input = mo.ui.number(
        start=1,
        stop=128,
        value=25,
        step=1,
        label="Sampling radius (mm)",
    )

    rotation_bounds_input = mo.ui.text(
        value="30,30,30",
        label="Euler bounds deg (+/- per axis): x,y,z",
    )

    repeats_input = mo.ui.number(
        start=1,
        stop=20,
        value=3,
        step=1,
        label="Benchmark repeats",
    )

    run_button = mo.ui.run_button(label="Run comparison")

    mo.vstack(
        [
            path_input,
            mo.hstack([modality_input, seed_input]),
            center_input,
            mo.hstack([base_patch_size_input, n_patches_input, radius_input]),
            rotation_bounds_input,
            repeats_input,
            run_button,
        ]
    )

    return (
        base_patch_size_input,
        center_input,
        modality_input,
        n_patches_input,
        path_input,
        radius_input,
        repeats_input,
        rotation_bounds_input,
        run_button,
        seed_input,
    )


@app.cell
def _(
    base_patch_size_input,
    center_input,
    modality_input,
    n_patches_input,
    parse_triplet,
    path_input,
    radius_input,
    repeats_input,
    rotation_bounds_input,
    run_button,
    seed_input,
):
    if not run_button.value:
        return None, None, None, None, None, None, None, None

    if not path_input.value.strip():
        raise ValueError("Please provide a scan path.")

    seed = int(seed_input.value)
    rng = np.random.default_rng(seed)

    bounds = np.asarray(parse_triplet(rotation_bounds_input.value, cast=float), dtype=np.float64)
    rotation = tuple(rng.uniform(low=-bounds, high=bounds).astype(np.float64))

    scan = nifti_scan(
        path_to_scan=path_input.value.strip(),
        median=0.0,
        stdev=1.0,
        base_patch_size=int(base_patch_size_input.value),
        modality=modality_input.value,
    )

    if center_input.value.strip():
        subset_center = np.asarray(parse_triplet(center_input.value, cast=int), dtype=np.int64)
    else:
        subset_center = scan.get_random_center_idx(float(radius_input.value), scan.patch_shape)

    common_kwargs = dict(
        n_patches=int(n_patches_input.value),
        subset_center=tuple(int(v) for v in subset_center.tolist()),
        sampling_radius_mm=float(radius_input.value),
        rotation_degrees=rotation,
        seed=seed,
    )

    def _timed(method: str):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = scan.train_sample(method=method, **common_kwargs)
        dt = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, dt, peak

    naive_result, naive_time, naive_peak = _timed("naive_full_rotate")
    opt_result, opt_time, opt_peak = _timed("optimized_local")

    perf_rows = []
    for method in ("naive_full_rotate", "optimized_local", "optimized_fused"):
        times = []
        peaks = []
        for trial in range(int(repeats_input.value)):
            trial_seed = seed + trial
            kwargs = dict(common_kwargs)
            kwargs["seed"] = trial_seed
            tracemalloc.start()
            t0 = time.perf_counter()
            _ = scan.train_sample(method=method, **kwargs)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            times.append(elapsed)
            peaks.append(peak)

        perf_rows.append(
            {
                "method": method,
                "median_seconds": float(np.median(times)),
                "median_peak_bytes": float(np.median(peaks)),
            }
        )

    perf_df = perf_rows

    return naive_result, naive_time, naive_peak, opt_result, opt_time, opt_peak, perf_df, scan


@app.cell
def _(naive_result, opt_result):
    if naive_result is None or opt_result is None:
        return None, None, None, None

    naive_p = np.asarray(naive_result["normalized_patches_3d"], dtype=np.float32)
    opt_p = np.asarray(opt_result["normalized_patches_3d"], dtype=np.float32)

    if naive_p.shape != opt_p.shape:
        raise ValueError(f"Patch shape mismatch: {naive_p.shape} vs {opt_p.shape}")

    patch_mae = float(np.mean(np.abs(naive_p - opt_p)))
    patch_max = float(np.max(np.abs(naive_p - opt_p)))

    naive_rel = np.asarray(naive_result["relative_patch_centers_pt"], dtype=np.float64)
    opt_rel = np.asarray(opt_result["relative_patch_centers_pt"], dtype=np.float64)

    rel_err = np.linalg.norm(naive_rel - opt_rel, axis=1)
    rel_mm_mae = float(np.mean(rel_err))
    rel_mm_max = float(np.max(rel_err))

    return patch_mae, patch_max, rel_mm_mae, rel_mm_max


@app.cell(hide_code=True)
def _(
    naive_peak,
    naive_result,
    naive_time,
    opt_peak,
    opt_result,
    opt_time,
    patch_mae,
    patch_max,
    perf_df,
    rel_mm_mae,
    rel_mm_max,
    scan,
):
    if naive_result is None:
        mo.md("Run the comparison to view metrics.")
        return

    geo = scan.rotation_sampler.geometry

    summary = {
        "orientation_codes_ras": geo.orientation_codes_ras,
        "spacing_mm": [float(v) for v in geo.spacing_mm.tolist()],
        "shape_vox": list(int(v) for v in geo.shape_vox),
        "modality": geo.modality,
        "method_naive": naive_result["method"],
        "method_optimized": opt_result["method"],
    }

    metric_rows = [
        {"metric": "Patch MAE (normalized)", "value": patch_mae, "target": "<= 0.01"},
        {"metric": "Patch Max Abs Error", "value": patch_max, "target": "monitor"},
        {"metric": "Relative Position Error MAE (mm)", "value": rel_mm_mae, "target": "<= 0.5"},
        {"metric": "Relative Position Error Max (mm)", "value": rel_mm_max, "target": "<= 0.5"},
        {"metric": "Naive Runtime (s)", "value": float(naive_time), "target": "report"},
        {"metric": "Optimized Runtime (s)", "value": float(opt_time), "target": "report"},
        {"metric": "Naive Peak Memory (bytes)", "value": float(naive_peak), "target": "report"},
        {"metric": "Optimized Peak Memory (bytes)", "value": float(opt_peak), "target": "report"},
    ]

    mo.vstack(
        [
            mo.md("### RAS Geometry Summary"),
            mo.ui.table(summary),
            mo.md("### Validation Metrics"),
            mo.ui.table(metric_rows),
            mo.md("### Benchmark (median across repeats)"),
            mo.ui.table(perf_df),
        ]
    )
    return


@app.cell
def _(naive_result, normalize_to_01, scan):
    if naive_result is None:
        return None, None

    scan_data = normalize_to_01(scan.nii_data)
    rgb = np.stack([scan_data, scan_data, scan_data], axis=-1)

    center = np.asarray(naive_result["subset_center"], dtype=np.int64)
    centers = np.asarray(naive_result["patch_centers_vox"], dtype=np.int64)

    rgb[
        max(center[0] - 2, 0):center[0] + 3,
        max(center[1] - 2, 0):center[1] + 3,
        max(center[2] - 2, 0):center[2] + 3,
        :,
    ] = [1.0, 0.0, 0.0]

    for c in centers:
        rgb[
            max(c[0] - 1, 0):c[0] + 2,
            max(c[1] - 1, 0):c[1] + 2,
            max(c[2] - 1, 0):c[2] + 2,
            :,
        ] = [0.0, 1.0, 0.0]

    axis = int(np.argmin(scan.patch_shape))
    start_idx = int(center[axis])
    slider = mo.ui.slider(start=0, stop=int(scan.nii_data.shape[axis] - 1), value=start_idx)
    slider

    return axis, rgb, slider


@app.cell
def _(axis, rgb, slider):
    if rgb is None:
        return

    slicer = [slice(None)] * 3
    slicer[axis] = int(slider.value)
    view = rgb[tuple(slicer)]
    mo.image(view)
    return


@app.cell
def _(naive_result, opt_result):
    if naive_result is None:
        return None

    patches_naive = np.asarray(naive_result["normalized_patches_3d"], dtype=np.float32)
    patches_opt = np.asarray(opt_result["normalized_patches_3d"], dtype=np.float32)

    patch_slider = mo.ui.slider(start=0, stop=int(patches_naive.shape[0] - 1), value=0)
    patch_slider

    return patch_slider, patches_naive, patches_opt


@app.cell
def _(patch_slider, patches_naive, patches_opt):
    if patch_slider is None:
        return

    idx = int(patch_slider.value)
    naive = patches_naive[idx]
    opt = patches_opt[idx]
    diff = np.abs(naive - opt)

    if naive.shape[0] == 1:
        naive_img = naive[0]
        opt_img = opt[0]
        diff_img = diff[0]
    else:
        mid = naive.shape[0] // 2
        naive_img = naive[mid]
        opt_img = opt[mid]
        diff_img = diff[mid]

    mo.hstack(
        [
            mo.vstack([mo.md("Naive"), mo.image(naive_img, width=180)]),
            mo.vstack([mo.md("Optimized"), mo.image(opt_img, width=180)]),
            mo.vstack([mo.md("Abs Diff"), mo.image(diff_img, width=180)]),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
