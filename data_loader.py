import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")

with app.setup:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, IterableDataset
    from math import pi
    from einops import rearrange, repeat
    import numpy as np
    import os
    import random
    import marimo as mo
    import zarr
    import pandas as pd
    import fastparquet
    import matplotlib.pyplot as plt
    from typing import Optional, Tuple, Dict, Any
    import pydicom
    # from rvt_model import PosEmbedding3D
    import nibabel as nib
    from skimage.transform import resize
    from scipy.spatial.transform import Rotation
    from scipy import ndimage


@app.cell(hide_code=True)
def _():
    metadata_df = pd.read_parquet("/cbica/home/gangarav/rsna25/aneurysm_labels_with_nifti_coords.parquet")
    table = mo.ui.table(
        metadata_df,
        selection="single",
    )
    table
    return metadata_df, table


@app.cell(hide_code=True)
def _():
    base_patch_shape = mo.ui.text(value="16", label="Patch Size (mm):")
    n_patches_input = mo.ui.text(value="64", label="Num Patches:")

    run_button = mo.ui.run_button()
    mo.vstack([base_patch_shape, n_patches_input, run_button])
    return base_patch_shape, n_patches_input, run_button


@app.cell(hide_code=True)
def _(base_patch_shape, metadata_df, n_patches_input, run_button, table):
    if run_button.value:
        base_patch_size = int(base_patch_shape.value)
        n_patches = int(n_patches_input.value)

        if table.value.empty:
            metadata_row = metadata_df.sample(1)
        else:
            metadata_row = table.value

        nifti_name = (
            metadata_row.iloc[0]["zarr_path"]
            .replace(
                "/cbica/home/gangarav/data_25_processed/",
                "/cbica/home/gangarav/rsna_any/nifti/",
            )
            .replace(".zarr", ".nii")
        )


        scan = nifti_scan(path_to_scan=nifti_name, median=metadata_row["median"].values[0], stdev=metadata_row["stdev"].values[0], base_patch_size=base_patch_size)
        patch_shape = scan.patch_shape
    return metadata_row, n_patches, nifti_name, patch_shape, scan


@app.cell
def _(n_patches, scan):
    result = scan.train_sample(
        n_patches,
        sampling_radius_mm=10,
    )
    result
    return


@app.cell
def _():
    center_point_vox_input = mo.ui.text(value="", label="Center Point Voxel (x,y,z) - leave blank for random:")
    sampling_radius_mm_input = mo.ui.text(value="", label="Sampling Radius (mm) - leave blank for random:")
    num_patches_input = mo.ui.text(value="64", label="Number of Patches:")
    voxel_spacing_input = mo.ui.text(value="", label="Voxel Spacing (x,y,z) - leave blank for auto:")
    rotation_input = mo.ui.text(value="", label="Rotation (x,y,z degrees) - leave blank for random:")

    generate_params_button = mo.ui.run_button(label="Generate Parameters")
    mo.vstack([center_point_vox_input, sampling_radius_mm_input, num_patches_input, voxel_spacing_input, rotation_input, generate_params_button])
    return (
        center_point_vox_input,
        generate_params_button,
        num_patches_input,
        rotation_input,
        sampling_radius_mm_input,
    )


@app.cell
def _(
    center_point_vox_input,
    generate_params_button,
    num_patches_input,
    rotation_input,
    sampling_radius_mm_input,
    scan,
):
    if generate_params_button.value:
        if center_point_vox_input.value.strip():
            center_point_vox = np.array([int(x) for x in center_point_vox_input.value.split(',')])
        else:
            center_point_vox = np.array([random.randint(0, s-1) for s in scan.nii_data.shape])

        if sampling_radius_mm_input.value.strip():
            sampling_radius_mm = float(sampling_radius_mm_input.value)
        else:
            sampling_radius_mm = random.uniform(20.0, 30.0)

        num_patches = int(num_patches_input.value)

        if rotation_input.value.strip():
            rotation_xyz_degrees = tuple(float(x) for x in rotation_input.value.split(','))
        else:
            rotation_xyz_degrees = (random.randint(-30, 30), random.randint(-30, 30), random.randint(-30, 30))
    return (
        center_point_vox,
        num_patches,
        rotation_xyz_degrees,
        sampling_radius_mm,
    )


@app.cell
def _(
    center_point_vox,
    num_patches,
    patch_shape,
    rotation_xyz_degrees,
    sampling_radius_mm,
    scan,
):
    # --- Step 0: Get instance properties ---
    affine = np.array(scan.nii_image.affine)
    voxel_spacing = np.array(scan.nii_image.header.get_zooms()[:3])

    # --- Step 1: Sample patch center locations in original voxel space ---
    patch_centers_vox = scan.sample_patch_centers(
        center_point_vox=center_point_vox,
        sampling_radius_mm=sampling_radius_mm,
        num_patches=num_patches,
        voxel_spacing=voxel_spacing,
        volume_shape=scan.nii_data.shape
    )

    # --- Step 2: Convert key center points from voxel to patient space (mm) --
    main_center_patient = scan.convert_voxel_to_patient(center_point_vox, affine)
    patch_centers_patient = scan.convert_voxel_to_patient(patch_centers_vox, affine)

    # --- Step 3: Pre-calculate all shapes, matrices, and other parameters ---
    params = scan.calculate_extraction_parameters(
        patch_shape_mm=patch_shape,
        rotation_xyz_degrees=rotation_xyz_degrees,
        voxel_spacing=voxel_spacing
    )

    # --- Step 4: Loop through centers to extract and process each patch ---
    final_patches = np.array([
        scan.extract_single_patch(
            data_volume=scan.nii_data,
            center_orig_vox=center_vox,
            params=params
        ) for center_vox in patch_centers_vox
    ])

    final_patches = np.squeeze(final_patches)

    # --- Step 5: Calculate the final rotated relative position vectors ---
    relative_rotated_centers = scan.calculate_rotated_relative_positions(
        patch_centers_patient=patch_centers_patient,
        main_center_patient=main_center_patient,
        forward_rotation_matrix=params["inverse_rotation_matrix"]
    )
    return (
        final_patches,
        main_center_patient,
        patch_centers_patient,
        patch_centers_vox,
        relative_rotated_centers,
    )


@app.cell
def _(
    center_point_vox,
    main_center_patient,
    patch_centers_patient,
    patch_centers_vox,
    relative_rotated_centers,
    scan,
):
    rgb = 128 + 10*((patch_centers_patient - main_center_patient) - relative_rotated_centers)

    scan_max = np.max(scan.nii_data)
    scan_min = np.min(scan.nii_data)
    scan_data = scan.normalize_pixels_to_range(scan.nii_data[:], scan_min, scan_max, (0, 1))

    scan_data_copy = np.stack([scan_data, scan_data, scan_data], axis=-1)

    scan_data_copy[center_point_vox[0]-2:center_point_vox[0]+3,
                  center_point_vox[1]-2:center_point_vox[1]+3,
                  center_point_vox[2]-2:center_point_vox[2]+3, :] = [1, 0, 0]

    for i, center in enumerate(patch_centers_vox):
        color = rgb[i] if i < len(rgb) else [0, 1, 0]
        scan_data_copy[center[0]-1:center[0]+2,
                       center[1]-1:center[1]+2,
                       center[2]-1:center[2]+2, :] = color/255
    return scan_data_copy, scan_max, scan_min


@app.cell
def _(final_patches):
    sld = mo.ui.slider(start=0, stop=final_patches.shape[0], value=final_patches.shape[0])
    sld
    return (sld,)


@app.cell
def _(final_patches, scan_max, scan_min, sld):
    patch_display = final_patches[sld.value, :, :].copy()
    patch_display[0, 0] = scan_max
    patch_display[-1, -1] = scan_min
    mo.image(src=patch_display, width=128)
    return


@app.cell
def _(
    main_center_patient,
    patch_centers_patient,
    relative_rotated_centers,
    sld,
):
    (relative_rotated_centers - (patch_centers_patient - main_center_patient))[sld.value]
    return


@app.cell
def _(main_center_patient, patch_centers_patient, sld):
    (patch_centers_patient - main_center_patient)[sld.value]
    return


@app.cell
def _(relative_rotated_centers, sld):
    relative_rotated_centers[sld.value]
    return


@app.cell
def _(center_point_vox, final_patches, patch_centers_vox, scan, sld):
    slice_axis = np.argmin(scan.patch_shape)

    # Determine the initial value for the slider based on the dynamic slice_axis
    if sld.value == final_patches.shape[0]:
        # Use the coordinate of the main prism center along the sliced axis
        slider_value = center_point_vox[slice_axis]
    else:
        # Use the coordinate of the selected patch center along the sliced axis
        slider_value = patch_centers_vox[sld.value][slice_axis]


    # Configure the slider to move along the correct axis of the full scan volume
    slider = mo.ui.slider(
        start=0,
        stop=scan.nii_data.shape[slice_axis] - 1, # Dynamically set the range
        value=slider_value                         # Dynamically set the initial position
    )

    slider
    return slice_axis, slider


@app.cell
def _(scan_data_copy, slice_axis, slider):
    slicer = [slice(None)] * 3 
    slicer[slice_axis] = slider.value
    slice_2d = scan_data_copy[tuple(slicer)]

    mo.image(src=slice_2d)
    return


@app.cell
def _(metadata_row):
    metadata_row
    return


@app.cell
def _(nifti_name):
    image = nib.as_closest_canonical(nib.load(nifti_name))
    _voxel_spacing = np.array(image.header.get_zooms()[:3])
    scan_size_vox = np.array(image.shape)
    scan_size_patient = scan_size_vox * _voxel_spacing

    most_different_axis = np.argmax(np.abs(_voxel_spacing - np.median(_voxel_spacing)))

    _voxel_spacing, scan_size_vox, scan_size_patient, most_different_axis
    return


@app.class_definition
class nifti_scan():

    def __init__(self, path_to_scan, median, stdev, base_patch_size):
        self.nii_image = nib.as_closest_canonical(nib.load(path_to_scan))
        self.nii_data = self.nii_image.get_fdata()
        self.med = median
        self.std = stdev
        self.patch_shape = self.get_outlier_axis_patch_shape(base_patch_size)

    def train_sample(
        self,
        n_patches: int,
        *, # Force subsequent arguments to be keyword-only for clarity
        subset_center: Optional[Tuple[int, int, int]] = None,
        sampling_radius_mm: Optional[int] = None,
        rotation_degrees: Optional[Tuple[int, int, int]] = None,
        wc: Optional[float] = None,
        ww: Optional[float] = None
    ) -> Dict[str, Any]:
        results = {}

        if wc is None or ww is None:
            wc, ww = self.get_random_wc_ww_for_scan()
        results['wc'], results['ww'] = wc, ww
        results['w_min'], results['w_max'] = wc - 0.5 * ww, wc + 0.5 * ww

        if sampling_radius_mm is None:
            sampling_radius_mm = random.uniform(20.0, 30.0)
        results['sampling_radius_mm'] = sampling_radius_mm

        if subset_center is None:
            subset_center = self.get_random_center_idx(sampling_radius_mm, self.patch_shape)
        results['subset_center'] = subset_center

        if rotation_degrees is None:
            rotation_degrees = (random.randint(-20, 20), random.randint(-20, 20), random.randint(-20, 20))
        results['rotation_degrees'] = rotation_degrees

        patches, prism_center_pt, patch_centers_vox, patch_centers_pt, relative_rotated_patch_centers_pt = self.sample_and_rotate_patches(
            n_patches,
            subset_center,
            sampling_radius_mm,
            self.patch_shape,
            rotation_degrees
        )

        results['prism_center_pt'] = prism_center_pt
        results['patch_centers_vox'] = patch_centers_vox 
        results['patch_centers_pt'] = patch_centers_pt   
        results['relative_patch_centers_pt'] = relative_rotated_patch_centers_pt
        results['raw_patches'] = patches

        normalized_patches = self.normalize_pixels_to_range(
            patches, results['w_min'], results['w_max']
        )

        results['normalized_patches'] = normalized_patches

        return results

    def get_outlier_axis_patch_shape(
        self, base_patch_size: int, similarity_threshold: float = 1.5
    ) -> Tuple[int, int, int]:
        """
        Determines patch shape by setting the scan's most distinct ("outlier") axis to 1.

        The outlier axis is the dimension that is furthest from the other two. It can
        be either the longest or the shortest. If all dimensions are similar, a random
        axis is chosen for slicing.

        Args:
            base_patch_size: The size for the two non-outlier axes (e.g., 16).
            similarity_threshold: The max/min ratio to consider dimensions similar.

        Returns:
            A (D, H, W) patch shape tuple.
        """
        shape = np.array(self.nii_data.shape)

        # 1. If dimensions are similar, pick a random axis for slicing
        if max(shape) / min(shape) < similarity_threshold:
            outlier_axis_idx = random.randint(0, 2)
        else:
            # 2. Find the outlier axis by finding the dimension that is furthest
            #    from the mean of the other two.
            diffs = [
                abs(shape[0] - (shape[1] + shape[2]) / 2.0), # How far axis 0 is from the mean of 1 and 2
                abs(shape[1] - (shape[0] + shape[2]) / 2.0), # How far axis 1 is from the mean of 0 and 2
                abs(shape[2] - (shape[0] + shape[1]) / 2.0)  # How far axis 2 is from the mean of 0 and 1
            ]
            outlier_axis_idx = np.argmax(diffs)

        # 3. Construct the patch shape
        patch_shape = [base_patch_size] * 3
        patch_shape[outlier_axis_idx] = 1

        return tuple(patch_shape)


    def get_random_wc_ww_for_scan(self):
        return random.uniform(self.med-self.std, self.med+self.std), random.uniform(2*self.std, 6*self.std)

    def get_random_center_idx(self, sampling_radius_mm, patch_shape_mm):
        """
        Finds a random center index using a simplified heuristic (1.5x patch shape)
        to define the boundary buffer.
        """
        voxel_spacing = np.array(self.nii_image.header.get_zooms()[:3])

        # --- 1. Calculate buffer for the patch using the 1.5x heuristic ---
        # Convert patch shape from mm to the original, anisotropic voxel units.
        patch_shape_vox = np.ceil(np.array(patch_shape_mm) / voxel_spacing)
        # The buffer is half of the patch size, scaled by 1.5 to approximate the diagonal.
        patch_buffer_vox = np.ceil(patch_shape_vox * 1.5 / 2.0).astype(int)

        # --- 2. Calculate buffer needed for the sampling radius ---
        sampling_radius_vox = np.ceil(sampling_radius_mm / voxel_spacing).astype(int)

        # --- 3. Combine buffers to define the safe sampling zone ---
        total_buffer_vox = patch_buffer_vox + sampling_radius_vox

        min_idx = total_buffer_vox
        max_idx = np.array(self.nii_data.shape) - total_buffer_vox - 1

        # Ensure we have a valid range to sample from.
        if np.any(max_idx < min_idx):
            raise ValueError(
                "Sampling radius and/or patch shape are too large for the given image dimensions."
            )

        # Generate a random index within the safe range.
        random_idx = np.array(
            [np.random.randint(low, high + 1) for low, high in zip(min_idx, max_idx)]
        )
        return random_idx

    def normalize_pixels_to_range(self, pixel_array, w_min, w_max, out_range=(-1.0, 1.0)):
        # Ensure w_max is greater than w_min to avoid division by zero
        if w_max <= w_min:
            w_max = w_min + 1e-6

        clipped_array = np.clip(pixel_array, w_min, w_max)
        scaled_01 = (clipped_array - w_min) / (w_max - w_min)
        out_min, out_max = out_range
        return scaled_01 * (out_max - out_min) + out_min

    @staticmethod
    def sample_patch_centers(
        center_point_vox: np.ndarray,
        sampling_radius_mm: float,
        num_patches: int,
        voxel_spacing: np.ndarray,
        volume_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Samples patch center coordinates within a sphere using rejection sampling.

        Returns:
            np.ndarray: An array of shape (num_patches, 3) with patch centers in voxel coordinates.
        """
        center_point_mm = np.array(center_point_vox) * voxel_spacing
        valid_centers_vox = []
        while len(valid_centers_vox) < num_patches:
            needed = num_patches - len(valid_centers_vox)
            points_to_sample = int(needed * 2.5) + 20 # Oversample to be safe

            # 1. Uniformly sample within a bounding box
            low_bound_mm = center_point_mm - sampling_radius_mm
            high_bound_mm = center_point_mm + sampling_radius_mm
            sampled_points_mm = np.random.uniform(
                low=low_bound_mm, high=high_bound_mm, size=(points_to_sample, 3)
            )

            # 2. Reject samples outside the sphere
            distances_sq = np.sum((sampled_points_mm - center_point_mm)**2, axis=1)
            in_sphere_mask = distances_sq <= sampling_radius_mm**2
            sampled_points_mm = sampled_points_mm[in_sphere_mask]

            # 3. Convert to voxel coordinates and reject if outside volume
            sampled_points_vox = np.round(sampled_points_mm / voxel_spacing).astype(int)
            in_volume_mask = np.all(sampled_points_vox >= 0, axis=1) & np.all(sampled_points_vox < volume_shape, axis=1)
            valid_centers_vox.extend(sampled_points_vox[in_volume_mask])

        return np.array(valid_centers_vox[:num_patches])

    @staticmethod
    def convert_voxel_to_patient(points_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Converts voxel coordinates to patient/world coordinates (mm)."""
        points_vox = np.atleast_2d(points_vox)
        homogeneous_coords = np.hstack((points_vox, np.ones((points_vox.shape[0], 1))))
        patient_coords = (affine @ homogeneous_coords.T).T[:, :3]
        return patient_coords.squeeze()

    @staticmethod
    def calculate_extraction_parameters(
        patch_shape_mm: Tuple[int, int, int],
        rotation_xyz_degrees: Tuple[float, float, float],
        voxel_spacing: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates all shapes, matrices, and factors needed for patch extraction."""
        final_patch_shape_iso_vox = np.ceil(patch_shape_mm).astype(int)

        rotation = Rotation.from_euler('xyz', rotation_xyz_degrees, degrees=True)
        forward_rotation_matrix = rotation.as_matrix()
        inverse_rotation_matrix = np.linalg.inv(forward_rotation_matrix)

        rotated_bbox_dims = np.abs(forward_rotation_matrix) @ final_patch_shape_iso_vox
        source_block_shape_iso_vox = np.ceil(rotated_bbox_dims).astype(int) + 3

        resampling_factor = voxel_spacing / np.array([1.0, 1.0, 1.0])
        source_block_shape_orig_vox = np.ceil(source_block_shape_iso_vox / resampling_factor).astype(int)

        return {
            "final_patch_shape_iso_vox": final_patch_shape_iso_vox,
            "source_block_shape_iso_vox": source_block_shape_iso_vox,
            "source_block_shape_orig_vox": source_block_shape_orig_vox,
            "forward_rotation_matrix": forward_rotation_matrix,
            "inverse_rotation_matrix": inverse_rotation_matrix,
        }

    @staticmethod
    def extract_single_patch(
        data_volume: np.ndarray,
        center_orig_vox: np.ndarray,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Extracts, resamples, rotates, and crops a single patch from the volume."""
        # --- 1. Safe Extraction from original volume ---
        starts = center_orig_vox - params['source_block_shape_orig_vox'] // 2
        ends = starts + params['source_block_shape_orig_vox']
        source_block = np.zeros(params['source_block_shape_orig_vox'], dtype=data_volume.dtype)

        crop_starts = np.maximum(starts, 0)
        crop_ends = np.minimum(ends, data_volume.shape)
        paste_starts = crop_starts - starts
        paste_ends = paste_starts + (crop_ends - crop_starts)

        source_block[tuple(slice(s, e) for s, e in zip(paste_starts, paste_ends))] = \
            data_volume[tuple(slice(s, e) for s, e in zip(crop_starts, crop_ends))]

        # --- 2. Resample to Isotropic ---
        zoom_factor = params['source_block_shape_iso_vox'] / source_block.shape
        isotropic_block = ndimage.zoom(source_block, zoom_factor, order=1, mode='constant', cval=0.0)

        # --- 3. Rotate ---
        block_center = (np.array(isotropic_block.shape) - 1) / 2.0
        offset = block_center - params['inverse_rotation_matrix'] @ block_center
        rotated_block = ndimage.affine_transform(isotropic_block, params['inverse_rotation_matrix'], offset=offset, order=1)

        # --- 4. Crop final patch from center ---
        rot_center = (np.array(rotated_block.shape) - 1) / 2.0
        crop_starts = np.round(rot_center - (np.array(params['final_patch_shape_iso_vox']) / 2.0)).astype(int)
        slicer = tuple(slice(s, s + d) for s, d in zip(crop_starts, params['final_patch_shape_iso_vox']))

        return rotated_block[slicer]

    @staticmethod
    def calculate_rotated_relative_positions(
        patch_centers_patient: np.ndarray,
        main_center_patient: np.ndarray,
        forward_rotation_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculates the final rotated relative position vectors."""
        relative_vectors = patch_centers_patient - main_center_patient
        return (forward_rotation_matrix @ relative_vectors.T).T

    def sample_and_rotate_patches(
        self,
        num_patches,
        center_point,
        sampling_radius_mm,
        patch_shape_mm,
        rotation_xyz_degrees=(0, 0, 0)
    ):
        """
        Orchestrates the patch extraction process using modular static helper functions.
        """
        # --- Step 0: Get instance properties ---
        affine = np.array(self.nii_image.affine)
        voxel_spacing = np.array(self.nii_image.header.get_zooms()[:3])

        # --- Step 1: Sample patch center locations in original voxel space ---
        patch_centers_vox = self.sample_patch_centers(
            center_point_vox=center_point,
            sampling_radius_mm=sampling_radius_mm,
            num_patches=num_patches,
            voxel_spacing=voxel_spacing,
            volume_shape=self.nii_data.shape
        )

        # --- Step 2: Convert key center points from voxel to patient space (mm) --
        main_center_patient = self.convert_voxel_to_patient(center_point, affine)
        patch_centers_patient = self.convert_voxel_to_patient(patch_centers_vox, affine)

        # --- Step 3: Pre-calculate all shapes, matrices, and other parameters ---
        params = self.calculate_extraction_parameters(
            patch_shape_mm=patch_shape_mm,
            rotation_xyz_degrees=rotation_xyz_degrees,
            voxel_spacing=voxel_spacing
        )

        # --- Step 4: Loop through centers to extract and process each patch ---
        final_patches = np.array([
            self.extract_single_patch(
                data_volume=self.nii_data,
                center_orig_vox=center_vox,
                params=params
            ) for center_vox in patch_centers_vox
        ])

        final_patches = np.squeeze(final_patches)

        # --- Step 5: Calculate the final rotated relative position vectors ---
        relative_rotated_centers = self.calculate_rotated_relative_positions(
            patch_centers_patient=patch_centers_patient,
            main_center_patient=main_center_patient,
            forward_rotation_matrix=params["inverse_rotation_matrix"]
        )

        # --- Step 6: Return all results ---
        return final_patches, main_center_patient, patch_centers_vox, patch_centers_patient, relative_rotated_centers


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
