import time
from pathlib import Path

import nibabel as nib
import numpy as np

from data_loader import nifti_scan


def _make_synthetic_phantom(shape, spacing, seed):
    """Create a smooth but structured 3D phantom with anisotropic spacing.

    The pattern intentionally mixes:
    - global low-frequency gradients
    - multiple ellipsoidal blobs
    - soft edges via tanh transitions

    This gives enough spatial structure to meaningfully test rotation/extraction
    while remaining stable under interpolation.
    """
    rng = np.random.default_rng(seed)

    x, y, z = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0]),
        np.linspace(-1.0, 1.0, shape[1]),
        np.linspace(-1.0, 1.0, shape[2]),
        indexing="ij",
    )

    # Baseline smooth field (roughly tissue-like intensity variation).
    volume = (
        60.0 * np.exp(-(x**2 + y**2 + z**2) / 0.55)
        + 20.0 * x
        - 12.0 * y
        + 6.0 * z
    )

    # Add several smooth structures with random centers/scales.
    for _ in range(6):
        cx, cy, cz = rng.uniform(-0.65, 0.65, size=3)
        sx, sy, sz = rng.uniform(0.12, 0.28, size=3)
        amplitude = rng.uniform(-30.0, 80.0)
        volume += amplitude * np.exp(
            -(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2 + ((z - cz) / sz) ** 2)
        )

    # Add gentle edge-like structures.
    volume += 15.0 * np.tanh(8.0 * (x + 0.25))
    volume -= 10.0 * np.tanh(7.0 * (z - 0.10))

    # Add low-amplitude noise to avoid degenerate perfect smoothness.
    volume += rng.normal(0.0, 0.8, size=shape)

    volume = volume.astype(np.float32)

    affine = np.eye(4, dtype=np.float64)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing
    affine[:3, 3] = np.array([11.0, -7.0, 25.0], dtype=np.float64)

    return volume, affine


def _build_scan(tmp_path: Path, case_name: str, shape, spacing, seed):
    volume, affine = _make_synthetic_phantom(shape=shape, spacing=spacing, seed=seed)
    scan_path = tmp_path / f"synthetic_{case_name}.nii.gz"
    nib.save(nib.Nifti1Image(volume, affine), str(scan_path))

    # For these synthetic tests we provide simple global stats.
    med = float(np.median(volume))
    std = float(np.std(volume))

    return nifti_scan(
        path_to_scan=str(scan_path),
        median=med,
        stdev=std,
        base_patch_size=16,
    )


def _pick_valid_center(scan: nifti_scan):
    """Pick a center/radius pair that is valid for current geometry."""
    for radius in (22.0, 18.0, 14.0, 10.0, 8.0):
        try:
            center = scan.get_random_center_idx(radius, scan.patch_shape)
            return center, radius
        except ValueError:
            continue
    raise RuntimeError("Could not find a valid center/radius for this synthetic scan.")


def _run_pair(scan: nifti_scan, center_vox, radius_mm, rotation_deg, seed, n_patches=32):
    """Run naive + optimized methods with identical center/patch-centers."""
    center_tuple = tuple(int(v) for v in np.asarray(center_vox, dtype=np.int64).tolist())

    common = dict(
        n_patches=int(n_patches),
        subset_center=center_tuple,
        sampling_radius_mm=float(radius_mm),
        rotation_degrees=tuple(float(v) for v in rotation_deg),
        seed=int(seed),
        wc=0.0,
        ww=250.0,
    )

    naive = scan.train_sample(method="naive_full_rotate", **common)
    centers = naive["patch_centers_vox"]

    fused = scan.train_sample(
        method="optimized_fused",
        patch_centers_vox=centers,
        **common,
    )

    local = scan.train_sample(
        method="optimized_local",
        patch_centers_vox=centers,
        **common,
    )

    return naive, fused, local


def test_position_fields_are_exact_and_definitionally_correct(tmp_path):
    """Validate position outputs independent of patch-intensity agreement."""
    shape = (88, 72, 48)
    spacing = (0.75, 0.75, 2.5)
    rotation = (15.0, 10.0, 5.0)

    scan = _build_scan(tmp_path, "positions", shape, spacing, seed=7)
    center, radius = _pick_valid_center(scan)

    naive, fused, _local = _run_pair(
        scan=scan,
        center_vox=center,
        radius_mm=radius,
        rotation_deg=rotation,
        seed=123,
        n_patches=36,
    )

    # 1) With shared patch centers, voxel centers must match exactly.
    np.testing.assert_array_equal(naive["patch_centers_vox"], fused["patch_centers_vox"])

    # 2) Unrotated and rotated relative vectors should match exactly between methods.
    np.testing.assert_allclose(
        naive["relative_patch_centers_pt"],
        fused["relative_patch_centers_pt"],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        naive["relative_patch_centers_pt_rotated"],
        fused["relative_patch_centers_pt_rotated"],
        atol=1e-7,
    )

    # 3) Definition check: relative = patch_center_world - prism_center_world.
    rel_expected = naive["patch_centers_pt"] - naive["prism_center_pt"]
    np.testing.assert_allclose(
        naive["relative_patch_centers_pt"],
        rel_expected,
        atol=1e-7,
    )

    # 4) Definition check: rotated_relative = R @ relative.
    R = naive["rotation_matrix_ras"]
    rel_rot_expected = (R @ rel_expected.T).T
    np.testing.assert_allclose(
        naive["relative_patch_centers_pt_rotated"],
        rel_rot_expected,
        atol=1e-7,
    )


def test_synthetic_randomized_patch_and_position_metrics_separately(tmp_path):
    """Stress test across many synthetic cases and random seeds.

    We track patch agreement and position agreement as separate metric groups.
    """
    cases = [
        ((96, 96, 64), (0.60, 0.60, 1.00)),
        ((88, 72, 48), (0.75, 0.75, 2.50)),
        ((80, 80, 40), (1.00, 1.00, 5.00)),
        ((88, 72, 48), (1.20, 0.90, 2.00)),
    ]
    rotations = [
        (15.0, 10.0, 5.0),
        (-20.0, 5.0, 12.0),
        (10.0, -12.0, -7.0),
        (0.0, 0.0, 0.0),
    ]

    fused_patch_mae = []
    local_patch_mae = []
    pos_unrot_max_err = []
    pos_rot_max_err = []
    speed_ratio_fused_to_naive = []

    successful_runs = 0

    for case_id, (shape, spacing) in enumerate(cases, start=1):
        scan = _build_scan(tmp_path, f"case_{case_id}", shape, spacing, seed=case_id)

        for rid, rotation in enumerate(rotations):
            center, radius = _pick_valid_center(scan)
            seed = 1000 + case_id * 10 + rid

            # Timed run for naive + fused (single-shot timing, enough for coarse regression gate).
            t0 = time.perf_counter()
            naive = scan.train_sample(
                n_patches=32,
                subset_center=tuple(int(v) for v in center.tolist()),
                sampling_radius_mm=radius,
                rotation_degrees=rotation,
                seed=seed,
                wc=0.0,
                ww=250.0,
                method="naive_full_rotate",
            )
            t1 = time.perf_counter()

            t2 = time.perf_counter()
            fused = scan.train_sample(
                n_patches=32,
                subset_center=tuple(int(v) for v in center.tolist()),
                sampling_radius_mm=radius,
                rotation_degrees=rotation,
                seed=seed,
                wc=0.0,
                ww=250.0,
                method="optimized_fused",
                patch_centers_vox=naive["patch_centers_vox"],
            )
            t3 = time.perf_counter()

            local = scan.train_sample(
                n_patches=32,
                subset_center=tuple(int(v) for v in center.tolist()),
                sampling_radius_mm=radius,
                rotation_degrees=rotation,
                seed=seed,
                wc=0.0,
                ww=250.0,
                method="optimized_local",
                patch_centers_vox=naive["patch_centers_vox"],
            )

            successful_runs += 1

            # --- Position metrics (separate from patch metrics) ---
            pos_unrot_max_err.append(
                float(
                    np.max(
                        np.abs(
                            naive["relative_patch_centers_pt"]
                            - fused["relative_patch_centers_pt"]
                        )
                    )
                )
            )
            pos_rot_max_err.append(
                float(
                    np.max(
                        np.abs(
                            naive["relative_patch_centers_pt_rotated"]
                            - fused["relative_patch_centers_pt_rotated"]
                        )
                    )
                )
            )

            # --- Patch metrics in normalized space ---
            fused_patch_mae.append(
                float(
                    np.mean(
                        np.abs(
                            naive["normalized_patches_3d"] - fused["normalized_patches_3d"]
                        )
                    )
                )
            )
            local_patch_mae.append(
                float(
                    np.mean(
                        np.abs(
                            naive["normalized_patches_3d"] - local["normalized_patches_3d"]
                        )
                    )
                )
            )

            naive_time = max(t1 - t0, 1e-9)
            fused_time = t3 - t2
            speed_ratio_fused_to_naive.append(float(fused_time / naive_time))

    # Ensure we actually exercised enough randomized cases.
    assert successful_runs >= 12

    # Position correctness should be effectively exact with shared centers/rotation.
    assert max(pos_unrot_max_err) < 1e-6
    assert max(pos_rot_max_err) < 1e-6

    # Patch quality gates: intentionally moderate because interpolation chains differ.
    fused_mae_arr = np.asarray(fused_patch_mae, dtype=np.float64)
    local_mae_arr = np.asarray(local_patch_mae, dtype=np.float64)

    assert float(np.mean(fused_mae_arr)) < 0.20
    assert float(np.percentile(fused_mae_arr, 95)) < 0.45

    assert float(np.mean(local_mae_arr)) < 0.22
    assert float(np.percentile(local_mae_arr, 95)) < 0.48

    # Fused path should be materially faster than naive in aggregate.
    ratio_arr = np.asarray(speed_ratio_fused_to_naive, dtype=np.float64)
    assert float(np.median(ratio_arr)) < 0.60
