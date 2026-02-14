"""
Explicit NIfTI data loader for prism-based self-supervised sampling.

This module intentionally favors clarity over compactness. It implements two
families of augmentation/extraction behavior:

1) Naive global rotation (`naive_full_rotate`)
   - Resample whole scan to isotropic spacing.
   - Rotate the whole isotropic scan around the *volume center*.
   - Extract axis-aligned patches in the rotated volume.

2) More efficient local methods (`optimized_local`, `optimized_fused`)
   - Avoid rotating the whole volume.
   - Extract each patch directly with a local transform.
   - Designed to represent the same global patient-rotation semantics while
     using less memory and compute.

Coordinate conventions
----------------------
- All scans are canonicalized to RAS+ at load time using
  `nib.as_closest_canonical(...)`.
- World/patient coordinates are always millimeters (mm).
- Rotation convention is SciPy `Rotation.from_euler("xyz", ...)` with lower-case
  axes. In SciPy this is EXTRINSIC fixed-axis X->Y->Z.

Returned position fields
------------------------
- `patch_centers_pt`: patch center positions in world mm (unrotated/original).
- `relative_patch_centers_pt`: patch center positions relative to prism center,
  in world mm (unrotated). This matches "patient-space relative center" semantics.
- `relative_patch_centers_pt_rotated`: same relative vectors after applying the
  augmentation rotation matrix.

Backwards compatibility note
----------------------------
Historically some code used `relative_patch_centers_pt` as rotated vectors. This
module now makes the unrotated meaning explicit and also returns
`relative_patch_centers_pt_rotated`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from scipy import ndimage
from scipy.spatial.transform import Rotation

SampleMethod = Literal["naive_full_rotate", "optimized_local", "optimized_fused", "naive_volume", "optimized_patch"]


def compute_robust_stats(data_array: np.ndarray) -> Tuple[float, float]:
    """Compute robust median/std after removing dominant repeated values.

    This follows your earlier heuristic so behavior is familiar:
    - remove a small set of most frequent values (often air/background)
    - compute median/std over the remaining voxels
    """
    data = np.asarray(data_array)
    stdev = np.std(data)
    values, counts = np.unique(data, return_counts=True)

    num_top_to_remove = int(np.maximum(stdev // 10, 1))
    if num_top_to_remove >= len(values):
        num_top_to_remove = max(len(values) - 1, 0)

    if num_top_to_remove == 0:
        return float(np.median(data)), float(np.std(data))

    top_indices = np.argsort(counts)[-num_top_to_remove:]
    mask = ~np.isin(data, values[top_indices])
    filtered = data[mask]

    if filtered.size == 0:
        return float(np.median(data)), float(np.std(data))

    return float(np.median(filtered)), float(np.std(filtered))


@dataclass
class _Geometry:
    """Minimal geometry payload exposed under `scan.rotation_sampler.geometry`.

    Several existing notebooks read this for quick summaries.
    """

    affine_ras: np.ndarray
    inv_affine_ras: np.ndarray
    spacing_mm: np.ndarray
    shape_vox: Tuple[int, int, int]
    orientation_codes_ras: Tuple[str, str, str]
    modality: str


class _RotationSamplerView:
    """Compatibility shim exposing geometry + robust normalization stats.

    This is intentionally small. It avoids forcing downstream code changes that
    referenced `scan.rotation_sampler.geometry`.
    """

    def __init__(self, geometry: _Geometry, robust_low: float, robust_high: float) -> None:
        self.geometry = geometry
        self._robust_low = float(robust_low)
        self._robust_high = float(robust_high)

    def get_scan_normalization_stats(self) -> Dict[str, Any]:
        return {
            "mode": "robust_percentile",
            "low": self._robust_low,
            "high": self._robust_high,
            "out_range": (-1.0, 1.0),
        }

    def normalize_with_scan_stats(self, values: np.ndarray) -> np.ndarray:
        return normalize_from_bounds(values, self._robust_low, self._robust_high)


def normalize_from_bounds(
    values: np.ndarray,
    low: float,
    high: float,
    out_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Linearly map values from [low, high] to `out_range`, with clipping."""
    if high <= low:
        high = low + 1e-6

    clipped = np.clip(values, low, high)
    scaled_01 = (clipped - low) / (high - low)
    out_min, out_max = out_range
    return scaled_01 * (out_max - out_min) + out_min


def _compute_robust_percentile_bounds(
    volume: np.ndarray,
    p_low: float,
    p_high: float,
) -> Tuple[float, float]:
    """Compute robust percentile bounds for normalization."""
    low, high = np.percentile(np.asarray(volume), [p_low, p_high])
    if high <= low:
        high = low + 1e-6
    return float(low), float(high)


class nifti_scan:
    """Scan loader + prism sampler with explicit geometry/rotation semantics.

    Public API intentionally mirrors your previous loader where possible.
    """

    def __init__(
        self,
        path_to_scan,
        median,
        stdev,
        base_patch_size,
        *,
        modality: str = "CT",
        robust_percentiles: Tuple[float, float] = (0.5, 99.5),
        similarity_threshold: float = 1.5,
    ):
        # Load and canonicalize to RAS+ so axis semantics are stable across scanners.
        self.nii_image = nib.as_closest_canonical(nib.load(path_to_scan))

        # Keep volume in float32 for memory/perf.
        self.nii_data = np.asarray(self.nii_image.get_fdata(), dtype=np.float32)

        # Store scalar stats used for random windowing fallback.
        self.med = float(median)
        self.std = float(stdev)

        # Modality is retained for metadata/reporting (CT/MR).
        self.modality = str(modality).upper()
        if self.modality not in {"CT", "MR"}:
            raise ValueError(f"Unsupported modality '{modality}'. Expected 'CT' or 'MR'.")

        # Core geometry in canonical RAS world frame.
        self.affine = np.asarray(self.nii_image.affine, dtype=np.float64)
        self.inv_affine = np.linalg.inv(self.affine)
        self.spacing = np.asarray(self.nii_image.header.get_zooms()[:3], dtype=np.float64)
        self.shape = np.asarray(self.nii_data.shape, dtype=np.int64)

        # Physical patch config:
        # - 16x16x1 pixels by definition
        # - non-outlier axes span `base_patch_size` mm
        # - outlier axis spans `base_patch_size / 16` mm
        self.base_patch_size = float(base_patch_size)
        self.target_spacing = self.base_patch_size / 16.0
        self.patch_shape = self.get_outlier_axis_patch_shape(
            int(base_patch_size),
            similarity_threshold=similarity_threshold,
        )

        # Robust normalization bounds used when explicit WC/WW is not passed.
        self.robust_percentiles = robust_percentiles
        self.robust_w_min, self.robust_w_max = _compute_robust_percentile_bounds(
            self.nii_data,
            robust_percentiles[0],
            robust_percentiles[1],
        )

        # Compatibility object for existing notebooks (`scan.rotation_sampler.geometry`).
        geometry = _Geometry(
            affine_ras=self.affine,
            inv_affine_ras=self.inv_affine,
            spacing_mm=self.spacing,
            shape_vox=tuple(int(v) for v in self.shape.tolist()),
            orientation_codes_ras=tuple(str(c) for c in aff2axcodes(self.affine)),
            modality=self.modality,
        )
        self.rotation_sampler = _RotationSamplerView(
            geometry,
            robust_low=self.robust_w_min,
            robust_high=self.robust_w_max,
        )

        # Cache for isotropic resampling, keyed by isotropic spacing.
        self._iso_cache: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # -------------------------------------------------------------------------
    # Basic utility methods (public API compatibility)
    # -------------------------------------------------------------------------

    def get_random_wc_ww_for_scan(self) -> Tuple[float, float]:
        return (
            random.uniform(self.med - self.std, self.med + self.std),
            random.uniform(2.0 * self.std, 6.0 * self.std),
        )

    def normalize_pixels_to_range(
        self,
        pixel_array: np.ndarray,
        w_min: float,
        w_max: float,
        out_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> np.ndarray:
        return normalize_from_bounds(pixel_array, float(w_min), float(w_max), out_range=out_range)

    @staticmethod
    def convert_voxel_to_patient(points_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Convert voxel coordinates to world/patient mm coordinates."""
        pts = np.atleast_2d(np.asarray(points_vox, dtype=np.float64))
        homo = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float64)))
        world = (np.asarray(affine, dtype=np.float64) @ homo.T).T[:, :3]
        return world.squeeze()

    @staticmethod
    def convert_patient_to_voxel(points_patient_mm: np.ndarray, inv_affine: np.ndarray) -> np.ndarray:
        """Convert world/patient mm coordinates to voxel coordinates."""
        pts = np.atleast_2d(np.asarray(points_patient_mm, dtype=np.float64))
        homo = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float64)))
        vox = (np.asarray(inv_affine, dtype=np.float64) @ homo.T).T[:, :3]
        return vox.squeeze()

    def get_outlier_axis_patch_shape(
        self,
        base_patch_size: int,
        similarity_threshold: float = 1.5,
    ) -> Tuple[float, float, float]:
        """Pick thin axis using *physical* dimensions (shape * spacing in mm).

        This is critical for MRI/CT mixtures where voxel counts alone are misleading.
        """
        physical_dims_mm = self.shape.astype(np.float64) * self.spacing

        if float(physical_dims_mm.max() / physical_dims_mm.min()) < float(similarity_threshold):
            # If dimensions are nearly isotropic at scan scale, no axis is obvious.
            # Keep old behavior: pick one axis randomly.
            outlier_axis_idx = random.randint(0, 2)
        else:
            diffs = np.asarray(
                [
                    abs(physical_dims_mm[0] - (physical_dims_mm[1] + physical_dims_mm[2]) / 2.0),
                    abs(physical_dims_mm[1] - (physical_dims_mm[0] + physical_dims_mm[2]) / 2.0),
                    abs(physical_dims_mm[2] - (physical_dims_mm[0] + physical_dims_mm[1]) / 2.0),
                ],
                dtype=np.float64,
            )
            outlier_axis_idx = int(np.argmax(diffs))

        patch_shape_mm = [float(base_patch_size)] * 3
        patch_shape_mm[outlier_axis_idx] = float(self.target_spacing)
        return tuple(patch_shape_mm)

    def get_random_center_idx(self, sampling_radius_mm, patch_shape_mm):
        """Sample a prism center with conservative spatial buffer from boundaries."""
        spacing = self.spacing

        patch_shape_vox = np.ceil(np.asarray(patch_shape_mm, dtype=np.float64) / spacing)

        # 1.5x heuristic is intentionally conservative for rotated local extraction.
        patch_buffer_vox = np.ceil(patch_shape_vox * 1.5 / 2.0).astype(np.int64)
        sampling_radius_vox = np.ceil(float(sampling_radius_mm) / spacing).astype(np.int64)

        total_buffer_vox = patch_buffer_vox + sampling_radius_vox
        min_idx = total_buffer_vox
        max_idx = self.shape - total_buffer_vox - 1

        if np.any(max_idx < min_idx):
            raise ValueError(
                "Sampling radius and/or patch shape are too large for the given image dimensions."
            )

        return np.asarray(
            [np.random.randint(low, high + 1) for low, high in zip(min_idx, max_idx)],
            dtype=np.int64,
        )

    @staticmethod
    def sample_patch_centers(
        center_point_vox: np.ndarray,
        sampling_radius_mm: float,
        num_patches: int,
        voxel_spacing: np.ndarray,
        volume_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """Backward-compatible sphere sampling helper using spacing-only conversion.

        For canonical RAS scans with near-diagonal affine this is usually fine.
        """
        center_vox = np.asarray(center_point_vox, dtype=np.float64)
        spacing = np.asarray(voxel_spacing, dtype=np.float64)
        center_mm = center_vox * spacing

        valid_centers: list[np.ndarray] = []
        while len(valid_centers) < int(num_patches):
            needed = int(num_patches) - len(valid_centers)
            sample_count = int(needed * 2.5) + 20

            low_mm = center_mm - float(sampling_radius_mm)
            high_mm = center_mm + float(sampling_radius_mm)
            sampled_mm = np.random.uniform(low=low_mm, high=high_mm, size=(sample_count, 3))

            dist_sq = np.sum((sampled_mm - center_mm) ** 2, axis=1)
            sampled_mm = sampled_mm[dist_sq <= float(sampling_radius_mm) ** 2]

            sampled_vox = np.rint(sampled_mm / spacing).astype(np.int64)
            in_volume = np.all(sampled_vox >= 0, axis=1) & np.all(
                sampled_vox < np.asarray(volume_shape, dtype=np.int64), axis=1
            )
            valid_centers.extend(sampled_vox[in_volume])

        return np.asarray(valid_centers[: int(num_patches)], dtype=np.int64)

    @staticmethod
    def calculate_rotated_relative_positions(
        patch_centers_patient: np.ndarray,
        main_center_patient: np.ndarray,
        forward_rotation_matrix: np.ndarray,
    ) -> np.ndarray:
        """Apply rotation to relative vectors (patch center - prism center)."""
        relative = np.asarray(patch_centers_patient, dtype=np.float64) - np.asarray(
            main_center_patient, dtype=np.float64
        )
        return (np.asarray(forward_rotation_matrix, dtype=np.float64) @ relative.T).T

    # -------------------------------------------------------------------------
    # Extraction parameterization + local patch extraction (debug-friendly)
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_extraction_parameters(
        patch_shape_mm: Tuple[float, float, float],
        rotation_xyz_degrees: Tuple[float, float, float],
        voxel_spacing: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute shapes/matrices for local per-patch extraction.

        Notes:
        - `target_iso_spacing` is derived from patch physical extent: max(mm)/16.
          This keeps 16 pixels on non-outlier axes and 1 on thin axis.
        - `resampling_factor` maps original voxel grid -> isotropic local grid.
        """
        patch_shape_mm = np.asarray(patch_shape_mm, dtype=np.float64)
        spacing = np.asarray(voxel_spacing, dtype=np.float64)

        target_iso_spacing = float(np.max(patch_shape_mm) / 16.0)
        final_patch_shape_iso_vox = np.maximum(
            np.ceil(patch_shape_mm / target_iso_spacing).astype(np.int64),
            1,
        )

        rotation = Rotation.from_euler("xyz", np.asarray(rotation_xyz_degrees, dtype=np.float64), degrees=True)
        forward_rotation_matrix = rotation.as_matrix()
        inverse_rotation_matrix = forward_rotation_matrix.T

        rotated_bbox_dims = np.abs(forward_rotation_matrix) @ final_patch_shape_iso_vox
        source_block_shape_iso_vox = np.maximum(
            np.ceil(rotated_bbox_dims).astype(np.int64),
            final_patch_shape_iso_vox,
        ) + 3

        # This factor means: orig_vox * resampling_factor -> iso_vox.
        resampling_factor = spacing / target_iso_spacing

        source_block_shape_orig_vox = np.ceil(
            source_block_shape_iso_vox / resampling_factor
        ).astype(np.int64)

        return {
            "target_iso_spacing": target_iso_spacing,
            "final_patch_shape_iso_vox": final_patch_shape_iso_vox,
            "source_block_shape_iso_vox": source_block_shape_iso_vox,
            "source_block_shape_orig_vox": source_block_shape_orig_vox,
            "resampling_factor": resampling_factor,
            "forward_rotation_matrix": forward_rotation_matrix,
            "inverse_rotation_matrix": inverse_rotation_matrix,
        }

    @staticmethod
    def extract_single_patch(
        data_volume: np.ndarray,
        center_orig_vox: np.ndarray,
        params: Dict[str, Any],
    ) -> np.ndarray:
        """Local extraction pipeline used by `optimized_local`.

        Pipeline:
        1) extract bounded source block from original anisotropic grid (with zero pad)
        2) resample source block to isotropic local grid
        3) rotate local isotropic block around *its own center*
        4) center crop final patch shape
        """
        center = np.asarray(center_orig_vox, dtype=np.int64)
        source_shape = np.asarray(params["source_block_shape_orig_vox"], dtype=np.int64)

        starts = center - source_shape // 2
        ends = starts + source_shape

        source_block = np.zeros(tuple(int(v) for v in source_shape.tolist()), dtype=np.asarray(data_volume).dtype)

        crop_starts = np.maximum(starts, 0)
        crop_ends = np.minimum(ends, np.asarray(data_volume).shape)
        paste_starts = crop_starts - starts
        paste_ends = paste_starts + (crop_ends - crop_starts)

        source_block[
            tuple(slice(int(s), int(e)) for s, e in zip(paste_starts, paste_ends))
        ] = np.asarray(data_volume)[
            tuple(slice(int(s), int(e)) for s, e in zip(crop_starts, crop_ends))
        ]

        isotropic_block = ndimage.zoom(
            source_block,
            np.asarray(params["resampling_factor"], dtype=np.float64),
            order=1,
            mode="constant",
            cval=0.0,
        )

        inv_rot = np.asarray(params["inverse_rotation_matrix"], dtype=np.float64)
        block_center = (np.asarray(isotropic_block.shape, dtype=np.float64) - 1.0) / 2.0
        offset = block_center - inv_rot @ block_center

        rotated_block = ndimage.affine_transform(
            isotropic_block,
            inv_rot,
            offset=offset,
            order=1,
            mode="constant",
            cval=0.0,
        )

        final_shape = np.asarray(params["final_patch_shape_iso_vox"], dtype=np.int64)
        rot_center = (np.asarray(rotated_block.shape, dtype=np.float64) - 1.0) / 2.0
        crop_starts = np.round(rot_center - (final_shape / 2.0)).astype(np.int64)

        slicer = tuple(slice(int(s), int(s + d)) for s, d in zip(crop_starts, final_shape))
        return rotated_block[slicer].astype(np.float32, copy=False)

    # -------------------------------------------------------------------------
    # Internal helpers for method-specific extraction
    # -------------------------------------------------------------------------

    @staticmethod
    def _canonical_method_name(method: str) -> str:
        method = str(method)
        aliases = {
            "naive_volume": "naive_full_rotate",
            "optimized_patch": "optimized_fused",
        }
        method = aliases.get(method, method)
        if method not in {"naive_full_rotate", "optimized_local", "optimized_fused"}:
            raise ValueError(f"Unknown sampling method '{method}'.")
        return method

    @staticmethod
    def _squeeze_spatial_singletons_keep_batch(patches_3d: np.ndarray) -> np.ndarray:
        """Remove singleton spatial axes but keep batch axis intact."""
        patches = np.asarray(patches_3d)
        if patches.ndim != 4:
            return patches

        squeeze_axes = [axis for axis in (1, 2, 3) if patches.shape[axis] == 1]
        if not squeeze_axes:
            return patches
        return np.squeeze(patches, axis=tuple(squeeze_axes))

    def _volume_center_vox(self) -> np.ndarray:
        return (self.shape.astype(np.float64) - 1.0) / 2.0

    def _volume_center_mm(self) -> np.ndarray:
        return np.asarray(self.convert_voxel_to_patient(self._volume_center_vox(), self.affine), dtype=np.float64)

    def _get_iso_volume(self, iso_spacing_mm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resample the whole scan to isotropic spacing and cache it.

        Returns:
        - iso_volume: float32 array
        - iso_affine: world mm per iso voxel
        - inv_iso_affine
        """
        spacing_key = float(iso_spacing_mm)
        if spacing_key in self._iso_cache:
            return self._iso_cache[spacing_key]

        zoom_factors = self.spacing / spacing_key
        iso_volume = ndimage.zoom(
            self.nii_data,
            zoom_factors,
            order=1,
            mode="constant",
            cval=0.0,
        ).astype(np.float32, copy=False)

        # Build affine that maps iso voxel coordinates -> world mm.
        # If x_iso = x_orig * zoom, then x_orig = x_iso / zoom.
        iso_affine = self.affine.copy()
        iso_affine[:3, :3] = self.affine[:3, :3] @ np.diag(1.0 / zoom_factors)
        inv_iso_affine = np.linalg.inv(iso_affine)

        self._iso_cache[spacing_key] = (iso_volume, iso_affine, inv_iso_affine)
        return self._iso_cache[spacing_key]

    @staticmethod
    def _sample_axis_aligned_patch_trilinear(
        volume: np.ndarray,
        center_vox: np.ndarray,
        patch_shape_vox: np.ndarray,
    ) -> np.ndarray:
        """Sample an axis-aligned patch with trilinear interpolation.

        `center_vox` can be fractional. This keeps naive and optimized extraction
        consistent when rotated centers land off-grid.
        """
        center = np.asarray(center_vox, dtype=np.float64).reshape(3)
        shape = np.asarray(patch_shape_vox, dtype=np.int64).reshape(3)

        offsets = [np.arange(int(d), dtype=np.float64) - (d - 1) / 2.0 for d in shape]
        grid = np.meshgrid(*offsets, indexing="ij")
        coords = [grid[ax].reshape(-1) + center[ax] for ax in range(3)]

        sampled = ndimage.map_coordinates(
            np.asarray(volume),
            coords,
            order=1,
            mode="constant",
            cval=0.0,
        )
        return sampled.reshape(tuple(int(v) for v in shape.tolist())).astype(np.float32, copy=False)

    def _sample_patch_centers_with_rng(
        self,
        center_point_vox: np.ndarray,
        sampling_radius_mm: float,
        num_patches: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample patch centers in a world-mm sphere, then map back to voxels.

        This uses full affine/inverse-affine transforms, not spacing-only math.
        """
        center_vox = np.asarray(center_point_vox, dtype=np.float64).reshape(3)
        center_mm = np.asarray(self.convert_voxel_to_patient(center_vox, self.affine), dtype=np.float64)

        valid: list[np.ndarray] = []
        seen = set()
        max_draws = 300000
        draws = 0

        shape = self.shape.astype(np.int64)
        radius = float(sampling_radius_mm)

        while len(valid) < int(num_patches):
            need = int(num_patches) - len(valid)
            sample_count = int(need * 3) + 32

            low_mm = center_mm - radius
            high_mm = center_mm + radius
            sampled_mm = rng.uniform(low=low_mm, high=high_mm, size=(sample_count, 3))
            draws += sample_count
            if draws > max_draws:
                raise ValueError(
                    "Could not sample enough valid patch centers. Reduce radius or patch size."
                )

            dist_sq = np.sum((sampled_mm - center_mm) ** 2, axis=1)
            sampled_mm = sampled_mm[dist_sq <= radius ** 2]
            if sampled_mm.size == 0:
                continue

            sampled_vox = np.rint(self.convert_patient_to_voxel(sampled_mm, self.inv_affine)).astype(np.int64)
            in_bounds = np.all(sampled_vox >= 0, axis=1) & np.all(sampled_vox < shape, axis=1)

            for c in sampled_vox[in_bounds]:
                key = (int(c[0]), int(c[1]), int(c[2]))
                if key in seen:
                    continue
                seen.add(key)
                valid.append(c.copy())
                if len(valid) == int(num_patches):
                    break

        return np.asarray(valid, dtype=np.int64)

    def _rotate_points_about_volume_center_mm(
        self,
        points_mm: np.ndarray,
        forward_rotation_matrix: np.ndarray,
    ) -> np.ndarray:
        """Rotate world points around the global volume center in world mm."""
        pts = np.atleast_2d(np.asarray(points_mm, dtype=np.float64))
        center_mm = self._volume_center_mm()
        rel = pts - center_mm
        rel_rot = (np.asarray(forward_rotation_matrix, dtype=np.float64) @ rel.T).T
        return rel_rot + center_mm

    def _rotate_iso_volume_about_its_center(
        self,
        iso_volume: np.ndarray,
        iso_affine: np.ndarray,
        inverse_rotation_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotate isotropic volume in voxel space around its geometric center.

        Returns:
        - rotated_iso_volume
        - rotated_iso_affine (output voxel -> world mm)
        - inverse rotated affine
        """
        iso_shape = np.asarray(iso_volume.shape, dtype=np.float64)
        center_iso_vox = (iso_shape - 1.0) / 2.0

        offset = center_iso_vox - inverse_rotation_matrix @ center_iso_vox
        rotated_iso = ndimage.affine_transform(
            iso_volume,
            inverse_rotation_matrix,
            offset=offset,
            order=1,
            mode="constant",
            cval=0.0,
        ).astype(np.float32, copy=False)

        # Build output affine from the actual transform used above:
        # input_vox = center + R_inv * (output_vox - center)
        # world = iso_affine * input_vox
        rot_affine = np.eye(4, dtype=np.float64)
        rot_affine[:3, :3] = iso_affine[:3, :3] @ inverse_rotation_matrix

        center_world = (iso_affine @ np.append(center_iso_vox, 1.0))[:3]
        rot_affine[:3, 3] = center_world - rot_affine[:3, :3] @ center_iso_vox

        inv_rot_affine = np.linalg.inv(rot_affine)
        return rotated_iso, rot_affine, inv_rot_affine

    def _extract_patches_naive_full_rotate(
        self,
        patch_centers_mm_unrot: np.ndarray,
        patch_shape_iso_vox: np.ndarray,
        forward_rotation_matrix: np.ndarray,
        inverse_rotation_matrix: np.ndarray,
        target_iso_spacing: float,
    ) -> np.ndarray:
        """Naive reference extraction (full-volume rotation around volume center)."""
        iso_volume, iso_affine, _ = self._get_iso_volume(target_iso_spacing)
        rotated_iso, _rot_affine, inv_rot_affine = self._rotate_iso_volume_about_its_center(
            iso_volume,
            iso_affine,
            inverse_rotation_matrix,
        )

        # In rotated world, patch centers also rotate around global volume center.
        patch_centers_mm_rot = self._rotate_points_about_volume_center_mm(
            patch_centers_mm_unrot,
            forward_rotation_matrix,
        )

        patches = []
        for center_mm_rot in patch_centers_mm_rot:
            center_rot_iso_vox = self.convert_patient_to_voxel(center_mm_rot, inv_rot_affine)
            patch = self._sample_axis_aligned_patch_trilinear(
                rotated_iso,
                center_rot_iso_vox,
                patch_shape_iso_vox,
            )
            patches.append(patch)

        return np.stack(patches, axis=0)

    def _extract_patches_optimized_fused(
        self,
        patch_centers_mm_unrot: np.ndarray,
        patch_shape_iso_vox: np.ndarray,
        inverse_rotation_matrix: np.ndarray,
        target_iso_spacing: float,
    ) -> np.ndarray:
        """Efficient direct extraction from original volume (no full-volume rotate).

        For each output patch voxel offset `u` in rotated frame (mm), we sample from:
            x_src_mm = center_unrot_mm + R_inv @ u

        This encodes global patient-rotation semantics while avoiding a full rotated
        volume allocation.
        """
        patch_shape = np.asarray(patch_shape_iso_vox, dtype=np.int64)

        offsets_mm = [
            (np.arange(int(d), dtype=np.float64) - (d - 1) / 2.0) * float(target_iso_spacing)
            for d in patch_shape
        ]
        mesh = np.meshgrid(*offsets_mm, indexing="ij")
        offset_vectors = np.stack([m.reshape(-1) for m in mesh], axis=1)

        patches = []
        for center_mm in np.atleast_2d(np.asarray(patch_centers_mm_unrot, dtype=np.float64)):
            src_mm = center_mm[None, :] + (inverse_rotation_matrix @ offset_vectors.T).T
            src_vox = self.convert_patient_to_voxel(src_mm, self.inv_affine)

            sampled = ndimage.map_coordinates(
                self.nii_data,
                [src_vox[:, axis] for axis in range(3)],
                order=1,
                mode="constant",
                cval=0.0,
            )
            patch = sampled.reshape(tuple(int(v) for v in patch_shape.tolist())).astype(np.float32, copy=False)
            patches.append(patch)

        return np.stack(patches, axis=0)

    def _extract_patches_optimized_local(
        self,
        patch_centers_vox: np.ndarray,
        params: Dict[str, Any],
    ) -> np.ndarray:
        """Local source-block extraction path (faster than full-volume naive)."""
        patches = [
            self.extract_single_patch(self.nii_data, center_orig_vox=center_vox, params=params)
            for center_vox in np.asarray(patch_centers_vox, dtype=np.int64)
        ]
        return np.stack(patches, axis=0)

    # -------------------------------------------------------------------------
    # Primary public APIs
    # -------------------------------------------------------------------------

    def sample_and_rotate_patches(
        self,
        num_patches,
        center_point,
        sampling_radius_mm,
        patch_shape_mm,
        rotation_xyz_degrees=(0, 0, 0),
        *,
        method: str = "optimized_fused",
        seed: Optional[int] = None,
        patch_centers_vox: Optional[np.ndarray] = None,
        return_debug: bool = False,
    ):
        """Core patch sampling/extraction routine.

        Returns tuple compatible with legacy code:
        (final_patches, prism_center_mm, patch_centers_vox, patch_centers_mm, relative_rotated_mm)

        If `return_debug=True`, also returns a `debug` dict with extra fields.
        """
        method = self._canonical_method_name(method)

        rng = np.random.default_rng(seed)

        center_point_vox = np.asarray(center_point, dtype=np.int64)
        rotation_xyz_degrees = tuple(float(v) for v in rotation_xyz_degrees)

        rotation = Rotation.from_euler("xyz", rotation_xyz_degrees, degrees=True)
        forward_rotation_matrix = rotation.as_matrix()
        inverse_rotation_matrix = forward_rotation_matrix.T

        params = self.calculate_extraction_parameters(
            patch_shape_mm=tuple(float(v) for v in patch_shape_mm),
            rotation_xyz_degrees=rotation_xyz_degrees,
            voxel_spacing=self.spacing,
        )
        patch_shape_iso_vox = np.asarray(params["final_patch_shape_iso_vox"], dtype=np.int64)
        target_iso_spacing = float(params["target_iso_spacing"])

        if patch_centers_vox is None:
            patch_centers_vox = self._sample_patch_centers_with_rng(
                center_point_vox,
                float(sampling_radius_mm),
                int(num_patches),
                rng,
            )
        else:
            patch_centers_vox = np.asarray(patch_centers_vox, dtype=np.int64)
            if patch_centers_vox.ndim != 2 or patch_centers_vox.shape[1] != 3:
                raise ValueError(
                    "patch_centers_vox must have shape (N, 3)."
                )
            if patch_centers_vox.shape[0] != int(num_patches):
                raise ValueError(
                    f"patch_centers_vox has {patch_centers_vox.shape[0]} centers but num_patches={num_patches}."
                )

        prism_center_mm = np.asarray(
            self.convert_voxel_to_patient(center_point_vox, self.affine),
            dtype=np.float64,
        )
        patch_centers_mm_unrot = np.asarray(
            self.convert_voxel_to_patient(patch_centers_vox, self.affine),
            dtype=np.float64,
        )

        # Relative vectors in patient/world frame (explicit requirement).
        relative_mm_unrot = patch_centers_mm_unrot - prism_center_mm

        # Rotated relative vectors in augmented frame.
        relative_mm_rot = (forward_rotation_matrix @ relative_mm_unrot.T).T

        # For global patient-rotation semantics, centers rotate around *volume center*.
        patch_centers_mm_rot = self._rotate_points_about_volume_center_mm(
            patch_centers_mm_unrot,
            forward_rotation_matrix,
        )
        prism_center_mm_rot = self._rotate_points_about_volume_center_mm(
            prism_center_mm,
            forward_rotation_matrix,
        ).reshape(3)

        if method == "naive_full_rotate":
            patches_3d = self._extract_patches_naive_full_rotate(
                patch_centers_mm_unrot,
                patch_shape_iso_vox,
                forward_rotation_matrix,
                inverse_rotation_matrix,
                target_iso_spacing,
            )
        elif method == "optimized_local":
            patches_3d = self._extract_patches_optimized_local(
                patch_centers_vox,
                params,
            )
        else:  # optimized_fused
            patches_3d = self._extract_patches_optimized_fused(
                patch_centers_mm_unrot,
                patch_shape_iso_vox,
                inverse_rotation_matrix,
                target_iso_spacing,
            )

        if not return_debug:
            return (
                patches_3d,
                prism_center_mm,
                patch_centers_vox,
                patch_centers_mm_unrot,
                relative_mm_rot,
            )

        debug = {
            "method": method,
            "rotation_matrix_ras": forward_rotation_matrix,
            "rotation_matrix_ras_inv": inverse_rotation_matrix,
            "patch_shape_iso_vox": patch_shape_iso_vox,
            "target_iso_spacing": target_iso_spacing,
            "patch_centers_pt_rotated": patch_centers_mm_rot,
            "prism_center_pt_rotated": prism_center_mm_rot,
            "relative_patch_centers_pt_unrot": relative_mm_unrot,
            "relative_patch_centers_pt_rotated": relative_mm_rot,
            "rotation_center_pt": self._volume_center_mm(),
            "params": params,
        }

        return (
            patches_3d,
            prism_center_mm,
            patch_centers_vox,
            patch_centers_mm_unrot,
            relative_mm_rot,
            debug,
        )

    def train_sample(
        self,
        n_patches: int,
        *,
        subset_center: Optional[Tuple[int, int, int]] = None,
        sampling_radius_mm: Optional[float] = None,
        rotation_degrees: Optional[Tuple[float, float, float]] = None,
        wc: Optional[float] = None,
        ww: Optional[float] = None,
        method: str = "optimized_fused",
        seed: Optional[int] = None,
        patch_centers_vox: Optional[np.ndarray] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """Sample one prism and return patches + position metadata.

        Args:
            method:
                - `naive_full_rotate` / `naive_volume`
                - `optimized_local`
                - `optimized_fused` / `optimized_patch`

        Returns keys include both unrotated and rotated position targets so you
        can choose explicitly in training/objective code.
        """
        results: Dict[str, Any] = {}

        if wc is None or ww is None:
            wc, ww = self.get_random_wc_ww_for_scan()
        wc = float(wc)
        ww = float(ww)
        results["wc"] = wc
        results["ww"] = ww
        results["w_min"] = wc - 0.5 * ww
        results["w_max"] = wc + 0.5 * ww

        if sampling_radius_mm is None:
            sampling_radius_mm = random.uniform(20.0, 30.0)
        sampling_radius_mm = float(sampling_radius_mm)
        results["sampling_radius_mm"] = sampling_radius_mm

        if subset_center is None:
            subset_center = tuple(
                int(v)
                for v in self.get_random_center_idx(
                    sampling_radius_mm,
                    self.patch_shape,
                ).tolist()
            )
        subset_center_arr = np.asarray(subset_center, dtype=np.int64)
        results["subset_center"] = subset_center_arr

        if rotation_degrees is None:
            rotation_degrees = (
                random.randint(-20, 20),
                random.randint(-20, 20),
                random.randint(-20, 20),
            )
        rotation_degrees = tuple(float(v) for v in rotation_degrees)
        results["rotation_degrees"] = rotation_degrees

        sample_out = self.sample_and_rotate_patches(
            num_patches=int(n_patches),
            center_point=subset_center_arr,
            sampling_radius_mm=sampling_radius_mm,
            patch_shape_mm=self.patch_shape,
            rotation_xyz_degrees=rotation_degrees,
            method=method,
            seed=seed,
            patch_centers_vox=patch_centers_vox,
            return_debug=True,
        )

        (
            patches_3d,
            prism_center_pt,
            patch_centers_vox_out,
            patch_centers_pt,
            _relative_rot_legacy,
            debug,
        ) = sample_out

        normalized_3d = self.normalize_pixels_to_range(
            patches_3d,
            results["w_min"],
            results["w_max"],
        )

        # Legacy 2D view (squeeze singleton spatial axis but keep batch dimension).
        raw_patches = self._squeeze_spatial_singletons_keep_batch(patches_3d)
        norm_patches = self._squeeze_spatial_singletons_keep_batch(normalized_3d)

        results["method"] = debug["method"]
        results["raw_patches_3d"] = patches_3d
        results["raw_patches"] = raw_patches
        results["normalized_patches_3d"] = normalized_3d
        results["normalized_patches"] = norm_patches

        # Explicit position outputs.
        results["subset_center_pt"] = prism_center_pt
        results["prism_center_pt"] = prism_center_pt
        results["prism_center_pt_rotated"] = debug["prism_center_pt_rotated"]
        results["patch_centers_vox"] = patch_centers_vox_out
        results["patch_centers_pt"] = patch_centers_pt
        results["patch_centers_pt_rotated"] = debug["patch_centers_pt_rotated"]

        # Primary position embedding requested by your spec: patient-space relative.
        results["relative_patch_centers_pt"] = debug["relative_patch_centers_pt_unrot"]

        # Also expose rotated-frame vectors for orientation-aware objectives.
        results["relative_patch_centers_pt_rotated"] = debug["relative_patch_centers_pt_rotated"]

        # Rotation metadata for prism-to-prism comparison heads.
        results["rotation_matrix_ras"] = debug["rotation_matrix_ras"]
        results["rotation_matrix_ras_inv"] = debug["rotation_matrix_ras_inv"]
        results["rotation_center_pt"] = debug["rotation_center_pt"]

        if return_debug:
            results["debug"] = debug

        return results
