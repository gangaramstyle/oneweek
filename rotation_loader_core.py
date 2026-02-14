from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from nibabel.processing import resample_to_output
from scipy import ndimage
from scipy.spatial.transform import Rotation

Modality = Literal["CT", "MR"]
SampleMethod = Literal["naive_full_rotate", "optimized_local", "optimized_fused"]


@dataclass(frozen=True)
class ScanGeometry:
    """Scanner and geometry priors for a canonicalized RAS scan."""

    affine_ras: np.ndarray
    inv_affine_ras: np.ndarray
    spacing_mm: np.ndarray
    shape_vox: Tuple[int, int, int]
    orientation_codes_ras: Tuple[str, str, str]
    modality: Modality


@dataclass(frozen=True)
class SamplingSpec:
    """Deterministic sampling and augmentation settings for one prism sample."""

    prism_center_vox: np.ndarray
    prism_size_mm: np.ndarray
    patch_size_mm: np.ndarray
    n_patches: int
    sampling_radius_mm: float
    rotation_euler_xyz_deg: np.ndarray
    seed: Optional[int] = None


@dataclass
class SampleBatch:
    """Output payload for one sampled prism world."""

    patches: np.ndarray
    patch_centers_mm: np.ndarray
    relative_patch_centers_mm_rot: np.ndarray
    prism_center_mm: np.ndarray
    rotation_euler_xyz_deg: np.ndarray
    rotation_matrix_ras: np.ndarray
    normalization_stats: Dict[str, Any]
    method: SampleMethod
    patch_centers_vox: np.ndarray
    patch_centers_mm_unrot: np.ndarray


def _as_float3(values: np.ndarray | Tuple[float, float, float], field_name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{field_name} must contain exactly 3 values, got {arr}")
    return arr


def _as_int3(values: np.ndarray | Tuple[int, int, int], field_name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{field_name} must contain exactly 3 values, got {arr}")
    return arr


def voxel_to_patient_mm(points_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points_vox = np.atleast_2d(np.asarray(points_vox, dtype=np.float64))
    homogeneous = np.hstack((points_vox, np.ones((points_vox.shape[0], 1), dtype=np.float64)))
    patient = (affine @ homogeneous.T).T[:, :3]
    return patient.squeeze()


def patient_mm_to_voxel(points_mm: np.ndarray, inv_affine: np.ndarray) -> np.ndarray:
    points_mm = np.atleast_2d(np.asarray(points_mm, dtype=np.float64))
    homogeneous = np.hstack((points_mm, np.ones((points_mm.shape[0], 1), dtype=np.float64)))
    vox = (inv_affine @ homogeneous.T).T[:, :3]
    return vox.squeeze()


def rotation_matrix_from_euler_xyz(rotation_euler_xyz_deg: np.ndarray) -> np.ndarray:
    euler = _as_float3(rotation_euler_xyz_deg, "rotation_euler_xyz_deg")
    # Extrinsic lowercase xyz around fixed world axes.
    return Rotation.from_euler("xyz", euler, degrees=True).as_matrix()


def compute_robust_scan_bounds(volume: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> Tuple[float, float]:
    low, high = np.percentile(volume, [p_low, p_high])
    if high <= low:
        high = low + 1e-6
    return float(low), float(high)


def normalize_from_bounds(
    values: np.ndarray,
    low: float,
    high: float,
    out_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    if high <= low:
        high = low + 1e-6
    clipped = np.clip(values, low, high)
    scaled_01 = (clipped - low) / (high - low)
    out_min, out_max = out_range
    return scaled_01 * (out_max - out_min) + out_min


def build_scan_geometry(nii_image: nib.Nifti1Image, modality: Modality) -> ScanGeometry:
    affine = np.asarray(nii_image.affine, dtype=np.float64)
    return ScanGeometry(
        affine_ras=affine,
        inv_affine_ras=np.linalg.inv(affine),
        spacing_mm=np.asarray(nii_image.header.get_zooms()[:3], dtype=np.float64),
        shape_vox=tuple(int(v) for v in nii_image.shape[:3]),
        orientation_codes_ras=tuple(str(c) for c in aff2axcodes(affine)),
        modality=modality,
    )


def _sample_axis_aligned_patch_trilinear(
    volume: np.ndarray,
    center_vox: np.ndarray,
    patch_shape_vox: np.ndarray,
) -> np.ndarray:
    center = _as_float3(center_vox, "center_vox")
    patch_shape = _as_int3(patch_shape_vox, "patch_shape_vox")

    offsets = [np.arange(int(d), dtype=np.float64) - (d - 1) / 2.0 for d in patch_shape]
    grid = np.meshgrid(*offsets, indexing="ij")
    coords = [g.reshape(-1) + center[axis] for axis, g in enumerate(grid)]

    sampled = ndimage.map_coordinates(
        volume,
        coords,
        order=1,
        mode="nearest",
    )
    return sampled.reshape(tuple(int(v) for v in patch_shape)).astype(np.float32, copy=False)


def _center_crop_3d(array: np.ndarray, target_shape: np.ndarray) -> np.ndarray:
    target = _as_int3(target_shape, "target_shape")
    starts = ((np.asarray(array.shape, dtype=np.int64) - target) // 2).astype(np.int64)
    if np.any(starts < 0):
        raise ValueError(
            f"Cannot center-crop shape {array.shape} to target {tuple(int(v) for v in target)}"
        )
    slices = tuple(slice(int(s), int(s + d)) for s, d in zip(starts, target))
    return array[slices]


def is_patch_in_bounds(center_vox: np.ndarray, patch_shape_vox: np.ndarray, volume_shape: Tuple[int, int, int]) -> bool:
    center = _as_float3(center_vox, "center_vox")
    patch_shape = _as_int3(patch_shape_vox, "patch_shape_vox")
    shape = np.asarray(volume_shape, dtype=np.float64)

    half_extent = (patch_shape - 1) / 2.0
    min_coords = center - half_extent
    max_coords = center + half_extent
    return bool(np.all(min_coords >= 0.0) and np.all(max_coords <= (shape - 1.0)))


def _calculate_local_extraction_parameters(
    patch_size_mm: np.ndarray,
    rotation_matrix: np.ndarray,
    spacing_mm: np.ndarray,
    iso_spacing_mm: float,
) -> Dict[str, np.ndarray]:
    patch_size_mm = _as_float3(patch_size_mm, "patch_size_mm")
    spacing_mm = _as_float3(spacing_mm, "spacing_mm")

    final_patch_shape_iso_vox = np.maximum(np.ceil(patch_size_mm / iso_spacing_mm).astype(np.int64), 1)
    rotated_bbox_dims = np.abs(rotation_matrix) @ final_patch_shape_iso_vox
    source_block_shape_iso_vox = np.ceil(rotated_bbox_dims).astype(np.int64) + 3

    source_block_shape_orig_vox = np.ceil(
        source_block_shape_iso_vox * iso_spacing_mm / spacing_mm
    ).astype(np.int64)

    return {
        "final_patch_shape_iso_vox": final_patch_shape_iso_vox,
        "source_block_shape_iso_vox": source_block_shape_iso_vox,
        "source_block_shape_orig_vox": source_block_shape_orig_vox,
    }


def _is_local_center_valid(
    center_vox: np.ndarray,
    source_block_shape_orig_vox: np.ndarray,
    volume_shape: Tuple[int, int, int],
) -> bool:
    center = _as_int3(center_vox, "center_vox")
    source_shape = _as_int3(source_block_shape_orig_vox, "source_block_shape_orig_vox")
    starts = center - source_shape // 2
    ends = starts + source_shape
    shape = np.asarray(volume_shape, dtype=np.int64)
    return bool(np.all(starts >= 0) and np.all(ends <= shape))


def _extract_rotated_patch_local(
    data_volume: np.ndarray,
    center_orig_vox: np.ndarray,
    params: Dict[str, np.ndarray],
    inverse_rotation_matrix: np.ndarray,
) -> np.ndarray:
    center = _as_int3(center_orig_vox, "center_orig_vox")
    source_shape = _as_int3(params["source_block_shape_orig_vox"], "source_block_shape_orig_vox")

    starts = center - source_shape // 2
    ends = starts + source_shape
    slices = tuple(slice(int(s), int(e)) for s, e in zip(starts, ends))

    source_block = data_volume[slices]
    if source_block.shape != tuple(int(v) for v in source_shape):
        raise ValueError(
            "Local extraction received an out-of-bounds center even though bounds validation should reject it."
        )

    source_iso_shape = _as_int3(params["source_block_shape_iso_vox"], "source_block_shape_iso_vox")
    zoom_factor = source_iso_shape / np.asarray(source_block.shape, dtype=np.float64)
    isotropic_block = ndimage.zoom(source_block, zoom_factor, order=1, mode="nearest")

    block_center = (np.asarray(isotropic_block.shape, dtype=np.float64) - 1.0) / 2.0
    offset = block_center - inverse_rotation_matrix @ block_center

    rotated_block = ndimage.affine_transform(
        isotropic_block,
        inverse_rotation_matrix,
        offset=offset,
        order=1,
        mode="nearest",
    )

    return _center_crop_3d(rotated_block, params["final_patch_shape_iso_vox"]).astype(np.float32, copy=False)


def _extract_rotated_patch_fused(
    data_volume: np.ndarray,
    patch_center_mm: np.ndarray,
    inverse_rotation_matrix: np.ndarray,
    geometry: ScanGeometry,
    patch_shape_iso_vox: np.ndarray,
    iso_spacing_mm: float,
) -> np.ndarray:
    patch_center_mm = _as_float3(patch_center_mm, "patch_center_mm")
    patch_shape = _as_int3(patch_shape_iso_vox, "patch_shape_iso_vox")

    offsets_mm = [
        (np.arange(int(d), dtype=np.float64) - (d - 1) / 2.0) * iso_spacing_mm
        for d in patch_shape
    ]
    offset_grid = np.meshgrid(*offsets_mm, indexing="ij")
    offset_vectors = np.stack([g.reshape(-1) for g in offset_grid], axis=1)

    source_mm = patch_center_mm[None, :] + (inverse_rotation_matrix @ offset_vectors.T).T
    source_vox = patient_mm_to_voxel(source_mm, geometry.inv_affine_ras)

    sampled = ndimage.map_coordinates(
        data_volume,
        [source_vox[:, axis] for axis in range(3)],
        order=1,
        mode="nearest",
    )

    return sampled.reshape(tuple(int(v) for v in patch_shape)).astype(np.float32, copy=False)


def _sample_patch_centers_in_sphere(
    center_point_vox: np.ndarray,
    sampling_radius_mm: float,
    num_patches: int,
    geometry: ScanGeometry,
    rng: np.random.Generator,
    validator: Callable[[np.ndarray], bool],
    max_candidate_draws: int = 300_000,
) -> np.ndarray:
    center_vox = _as_int3(center_point_vox, "center_point_vox")
    center_mm = voxel_to_patient_mm(center_vox, geometry.affine_ras)

    valid_centers: list[np.ndarray] = []
    seen: set[Tuple[int, int, int]] = set()
    draws = 0

    while len(valid_centers) < int(num_patches):
        needed = int(num_patches) - len(valid_centers)
        sample_count = int(needed * 3) + 32

        low_bound_mm = center_mm - sampling_radius_mm
        high_bound_mm = center_mm + sampling_radius_mm
        sampled_points_mm = rng.uniform(low=low_bound_mm, high=high_bound_mm, size=(sample_count, 3))
        draws += sample_count
        if draws > max_candidate_draws:
            raise ValueError(
                "Could not sample enough valid patch centers. "
                "Try reducing sampling radius, patch size, or rotation bounds."
            )

        distances_sq = np.sum((sampled_points_mm - center_mm) ** 2, axis=1)
        sampled_points_mm = sampled_points_mm[distances_sq <= sampling_radius_mm**2]
        if sampled_points_mm.size == 0:
            continue

        sampled_points_vox = np.rint(
            patient_mm_to_voxel(sampled_points_mm, geometry.inv_affine_ras)
        ).astype(np.int64)

        shape = np.asarray(geometry.shape_vox, dtype=np.int64)
        in_volume = np.all(sampled_points_vox >= 0, axis=1) & np.all(sampled_points_vox < shape, axis=1)

        for candidate in sampled_points_vox[in_volume]:
            key = (int(candidate[0]), int(candidate[1]), int(candidate[2]))
            if key in seen:
                continue
            if not validator(candidate):
                continue
            seen.add(key)
            valid_centers.append(candidate.copy())
            if len(valid_centers) == int(num_patches):
                break

    return np.asarray(valid_centers, dtype=np.int64)


class RotationAwarePrismSampler:
    """Reference and optimized prism samplers with shared geometry conventions."""

    def __init__(
        self,
        nii_image: nib.Nifti1Image,
        data_volume: np.ndarray,
        *,
        modality: Modality = "CT",
        iso_spacing_mm: float = 1.0,
        robust_percentiles: Tuple[float, float] = (0.5, 99.5),
    ) -> None:
        self.nii_image = nii_image
        self.data_volume = np.asarray(data_volume, dtype=np.float32)
        self.geometry = build_scan_geometry(nii_image, modality=modality)
        self.iso_spacing_mm = float(iso_spacing_mm)
        self.robust_percentiles = robust_percentiles

        low, high = compute_robust_scan_bounds(
            self.data_volume,
            p_low=robust_percentiles[0],
            p_high=robust_percentiles[1],
        )
        self._scan_norm_stats = {
            "mode": "robust_percentile",
            "p_low": float(robust_percentiles[0]),
            "p_high": float(robust_percentiles[1]),
            "low": low,
            "high": high,
            "out_range": (-1.0, 1.0),
        }
        self._iso_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def get_scan_normalization_stats(self) -> Dict[str, Any]:
        return dict(self._scan_norm_stats)

    def normalize_with_scan_stats(self, values: np.ndarray) -> np.ndarray:
        return normalize_from_bounds(values, self._scan_norm_stats["low"], self._scan_norm_stats["high"])

    @staticmethod
    def normalize_with_window(values: np.ndarray, wc: float, ww: float) -> np.ndarray:
        w_min = wc - 0.5 * ww
        w_max = wc + 0.5 * ww
        return normalize_from_bounds(values, w_min, w_max)

    def _get_iso_resampled_volume(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._iso_cache is None:
            image = nib.Nifti1Image(self.data_volume, self.geometry.affine_ras)
            iso_img = resample_to_output(
                image,
                voxel_sizes=(self.iso_spacing_mm, self.iso_spacing_mm, self.iso_spacing_mm),
                order=1,
                mode="nearest",
            )
            iso_data = np.asarray(iso_img.get_fdata(), dtype=np.float32)
            iso_affine = np.asarray(iso_img.affine, dtype=np.float64)
            inv_iso_affine = np.linalg.inv(iso_affine)
            self._iso_cache = (iso_data, iso_affine, inv_iso_affine)
        return self._iso_cache

    def _sample_naive_full_rotate(
        self,
        spec: SamplingSpec,
        rotation_matrix: np.ndarray,
        rng: np.random.Generator,
    ) -> SampleBatch:
        iso_volume, iso_affine, inv_iso_affine = self._get_iso_resampled_volume()

        prism_center_mm = voxel_to_patient_mm(spec.prism_center_vox, self.geometry.affine_ras)
        prism_center_iso_vox = patient_mm_to_voxel(prism_center_mm, inv_iso_affine)
        inverse_rotation_matrix = rotation_matrix.T

        offset = prism_center_iso_vox - inverse_rotation_matrix @ prism_center_iso_vox
        rotated_iso = ndimage.affine_transform(
            iso_volume,
            inverse_rotation_matrix,
            offset=offset,
            order=1,
            mode="nearest",
        )

        patch_shape_iso_vox = np.maximum(
            np.ceil(spec.patch_size_mm / self.iso_spacing_mm).astype(np.int64),
            1,
        )

        def validator(center_vox: np.ndarray) -> bool:
            center_mm = voxel_to_patient_mm(center_vox, self.geometry.affine_ras)
            rel = center_mm - prism_center_mm
            rotated_center_mm = prism_center_mm + (rotation_matrix @ rel)
            rotated_center_iso_vox = patient_mm_to_voxel(rotated_center_mm, inv_iso_affine)
            return is_patch_in_bounds(rotated_center_iso_vox, patch_shape_iso_vox, rotated_iso.shape)

        patch_centers_vox = _sample_patch_centers_in_sphere(
            center_point_vox=spec.prism_center_vox,
            sampling_radius_mm=spec.sampling_radius_mm,
            num_patches=spec.n_patches,
            geometry=self.geometry,
            rng=rng,
            validator=validator,
        )

        patch_centers_mm_unrot = voxel_to_patient_mm(patch_centers_vox, self.geometry.affine_ras)
        relative_mm = patch_centers_mm_unrot - prism_center_mm
        relative_rot_mm = (rotation_matrix @ relative_mm.T).T
        patch_centers_mm_rot = prism_center_mm + relative_rot_mm

        patches = []
        for rotated_center_mm in patch_centers_mm_rot:
            rotated_center_iso_vox = patient_mm_to_voxel(rotated_center_mm, inv_iso_affine)
            patch = _sample_axis_aligned_patch_trilinear(
                rotated_iso,
                rotated_center_iso_vox,
                patch_shape_iso_vox,
            )
            patches.append(patch)

        return SampleBatch(
            patches=np.stack(patches, axis=0),
            patch_centers_mm=patch_centers_mm_rot,
            relative_patch_centers_mm_rot=relative_rot_mm,
            prism_center_mm=prism_center_mm,
            rotation_euler_xyz_deg=np.asarray(spec.rotation_euler_xyz_deg, dtype=np.float64),
            rotation_matrix_ras=rotation_matrix,
            normalization_stats=self.get_scan_normalization_stats(),
            method="naive_full_rotate",
            patch_centers_vox=patch_centers_vox,
            patch_centers_mm_unrot=patch_centers_mm_unrot,
        )

    def _sample_optimized_local_or_fused(
        self,
        spec: SamplingSpec,
        rotation_matrix: np.ndarray,
        rng: np.random.Generator,
        method: SampleMethod,
    ) -> SampleBatch:
        params = _calculate_local_extraction_parameters(
            patch_size_mm=spec.patch_size_mm,
            rotation_matrix=rotation_matrix,
            spacing_mm=self.geometry.spacing_mm,
            iso_spacing_mm=self.iso_spacing_mm,
        )

        source_block_shape_orig_vox = params["source_block_shape_orig_vox"]

        def validator(center_vox: np.ndarray) -> bool:
            return _is_local_center_valid(center_vox, source_block_shape_orig_vox, self.geometry.shape_vox)

        patch_centers_vox = _sample_patch_centers_in_sphere(
            center_point_vox=spec.prism_center_vox,
            sampling_radius_mm=spec.sampling_radius_mm,
            num_patches=spec.n_patches,
            geometry=self.geometry,
            rng=rng,
            validator=validator,
        )

        prism_center_mm = voxel_to_patient_mm(spec.prism_center_vox, self.geometry.affine_ras)
        patch_centers_mm_unrot = voxel_to_patient_mm(patch_centers_vox, self.geometry.affine_ras)

        inverse_rotation_matrix = rotation_matrix.T
        patch_shape_iso_vox = params["final_patch_shape_iso_vox"]

        patches = []
        if method == "optimized_fused":
            for center_mm in patch_centers_mm_unrot:
                patch = _extract_rotated_patch_fused(
                    self.data_volume,
                    center_mm,
                    inverse_rotation_matrix,
                    self.geometry,
                    patch_shape_iso_vox,
                    self.iso_spacing_mm,
                )
                patches.append(patch)
        else:
            for center_vox in patch_centers_vox:
                patch = _extract_rotated_patch_local(
                    self.data_volume,
                    center_vox,
                    params,
                    inverse_rotation_matrix,
                )
                patches.append(patch)

        relative_mm = patch_centers_mm_unrot - prism_center_mm
        relative_rot_mm = (rotation_matrix @ relative_mm.T).T
        patch_centers_mm_rot = prism_center_mm + relative_rot_mm

        return SampleBatch(
            patches=np.stack(patches, axis=0),
            patch_centers_mm=patch_centers_mm_rot,
            relative_patch_centers_mm_rot=relative_rot_mm,
            prism_center_mm=prism_center_mm,
            rotation_euler_xyz_deg=np.asarray(spec.rotation_euler_xyz_deg, dtype=np.float64),
            rotation_matrix_ras=rotation_matrix,
            normalization_stats=self.get_scan_normalization_stats(),
            method=method,
            patch_centers_vox=patch_centers_vox,
            patch_centers_mm_unrot=patch_centers_mm_unrot,
        )

    def sample(self, spec: SamplingSpec, method: SampleMethod = "optimized_local") -> SampleBatch:
        method = str(method)
        if method not in {"naive_full_rotate", "optimized_local", "optimized_fused"}:
            raise ValueError(f"Unknown sampling method: {method}")

        rotation_matrix = rotation_matrix_from_euler_xyz(spec.rotation_euler_xyz_deg)
        rng = np.random.default_rng(spec.seed)

        if method == "naive_full_rotate":
            return self._sample_naive_full_rotate(spec, rotation_matrix, rng)

        return self._sample_optimized_local_or_fused(spec, rotation_matrix, rng, method=method)
