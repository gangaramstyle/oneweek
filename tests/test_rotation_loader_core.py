import numpy as np
import nibabel as nib
import pytest

from rotation_loader_core import (
    RotationAwarePrismSampler,
    SamplingSpec,
    patient_mm_to_voxel,
    voxel_to_patient_mm,
)


def _make_anisotropic_scan(shape=(64, 64, 28), spacing=(0.7, 0.7, 2.0)):
    # Smooth synthetic phantom with stable gradients and a bright landmark.
    grid = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0]),
        np.linspace(-1.0, 1.0, shape[1]),
        np.linspace(-1.0, 1.0, shape[2]),
        indexing="ij",
    )
    volume = (0.4 * grid[0] + 0.3 * grid[1] - 0.2 * grid[2]).astype(np.float32)
    volume += (np.exp(-(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) / 0.12) * 3.0).astype(np.float32)

    landmark = (int(shape[0] * 0.7), int(shape[1] * 0.45), int(shape[2] * 0.5))
    volume[landmark] += 4.0

    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    affine[:3, 3] = np.array([5.0, -12.0, 22.0], dtype=np.float64)

    img = nib.Nifti1Image(volume, affine)
    return img, volume


def _make_spec(center_vox, seed=1234):
    return SamplingSpec(
        prism_center_vox=np.asarray(center_vox, dtype=np.int64),
        prism_size_mm=np.asarray([1.0, 16.0, 16.0], dtype=np.float64),
        patch_size_mm=np.asarray([1.0, 16.0, 16.0], dtype=np.float64),
        n_patches=24,
        sampling_radius_mm=8.0,
        rotation_euler_xyz_deg=np.asarray([11.0, -7.0, 14.0], dtype=np.float64),
        seed=seed,
    )


def test_naive_vs_optimized_fused_within_error_gate():
    img, volume = _make_anisotropic_scan()
    sampler = RotationAwarePrismSampler(img, volume, modality="CT")

    spec = _make_spec(center_vox=(30, 30, 14), seed=77)

    naive = sampler.sample(spec, method="naive_full_rotate")
    opt = sampler.sample(spec, method="optimized_fused")

    assert naive.patches.shape == opt.patches.shape
    mae = float(np.mean(np.abs(naive.patches - opt.patches)))
    rel_err = np.linalg.norm(
        naive.relative_patch_centers_mm_rot - opt.relative_patch_centers_mm_rot,
        axis=1,
    )

    assert mae <= 0.01
    assert float(np.mean(rel_err)) <= 0.5


def test_optimized_local_is_shape_and_position_consistent():
    img, volume = _make_anisotropic_scan()
    sampler = RotationAwarePrismSampler(img, volume, modality="CT")

    spec = _make_spec(center_vox=(30, 30, 14), seed=77)
    naive = sampler.sample(spec, method="naive_full_rotate")
    local = sampler.sample(spec, method="optimized_local")

    assert naive.patches.shape == local.patches.shape
    np.testing.assert_array_equal(naive.patch_centers_vox, local.patch_centers_vox)
    np.testing.assert_allclose(
        naive.relative_patch_centers_mm_rot,
        local.relative_patch_centers_mm_rot,
        atol=1e-6,
    )


def test_seeded_reproducibility():
    img, volume = _make_anisotropic_scan()
    sampler = RotationAwarePrismSampler(img, volume, modality="MR")

    spec = _make_spec(center_vox=(28, 32, 13), seed=999)
    sample_a = sampler.sample(spec, method="optimized_fused")
    sample_b = sampler.sample(spec, method="optimized_fused")

    np.testing.assert_allclose(sample_a.patches, sample_b.patches, atol=1e-6)
    np.testing.assert_array_equal(sample_a.patch_centers_vox, sample_b.patch_centers_vox)
    np.testing.assert_allclose(
        sample_a.relative_patch_centers_mm_rot,
        sample_b.relative_patch_centers_mm_rot,
        atol=1e-6,
    )


def test_orientation_robustness_after_canonicalization():
    img_base, volume_base = _make_anisotropic_scan()

    # Create an equivalent scan with flipped X orientation (L<->R).
    flipped = volume_base[::-1, :, :].copy()
    affine_flip = img_base.affine.copy()
    affine_flip[0, 0] *= -1.0
    affine_flip[0, 3] += (volume_base.shape[0] - 1) * abs(img_base.affine[0, 0])
    img_flip = nib.Nifti1Image(flipped, affine_flip)

    base_canon = nib.as_closest_canonical(img_base)
    flip_canon = nib.as_closest_canonical(img_flip)

    sampler_a = RotationAwarePrismSampler(base_canon, np.asarray(base_canon.get_fdata(), dtype=np.float32), modality="CT")
    sampler_b = RotationAwarePrismSampler(flip_canon, np.asarray(flip_canon.get_fdata(), dtype=np.float32), modality="CT")

    center_mm = voxel_to_patient_mm(np.asarray([30, 30, 14]), sampler_a.geometry.affine_ras)
    center_a = np.rint(patient_mm_to_voxel(center_mm, sampler_a.geometry.inv_affine_ras)).astype(np.int64)
    center_b = np.rint(patient_mm_to_voxel(center_mm, sampler_b.geometry.inv_affine_ras)).astype(np.int64)

    spec_a = _make_spec(center_a, seed=2024)
    spec_b = _make_spec(center_b, seed=2024)

    out_a = sampler_a.sample(spec_a, method="optimized_fused")
    out_b = sampler_b.sample(spec_b, method="optimized_fused")

    # Canonical RAS + same world center/seed should produce near-identical rotated relative vectors.
    np.testing.assert_allclose(
        out_a.relative_patch_centers_mm_rot,
        out_b.relative_patch_centers_mm_rot,
        atol=1e-4,
    )


def test_rejects_invalid_boundary_configuration():
    img, volume = _make_anisotropic_scan(shape=(40, 40, 16))
    sampler = RotationAwarePrismSampler(img, volume, modality="CT")

    spec = SamplingSpec(
        prism_center_vox=np.asarray([2, 2, 2], dtype=np.int64),
        prism_size_mm=np.asarray([1.0, 20.0, 20.0], dtype=np.float64),
        patch_size_mm=np.asarray([1.0, 20.0, 20.0], dtype=np.float64),
        n_patches=16,
        sampling_radius_mm=16.0,
        rotation_euler_xyz_deg=np.asarray([20.0, 20.0, 20.0], dtype=np.float64),
        seed=1,
    )

    with pytest.raises(ValueError):
        sampler.sample(spec, method="optimized_local")
