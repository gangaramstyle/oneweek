import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _():
    import torch
    import torch.nn.functional as F
    import nibabel as nib
    import numpy as np
    import random
    from scipy.spatial.transform import Rotation
    from scipy import ndimage
    from typing import Tuple, Dict, Any, Optional, List
    import marimo as mo
    import pandas as pd
    import time
    from torch.utils.data.dataloader import default_collate
    from torch.utils.data import Dataset, DataLoader

    # Define device for GPU transformer
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo.md(f"**Device for validation:** `{DEVICE}`")
    return (
        Any,
        DEVICE,
        Dataset,
        Dict,
        F,
        List,
        Rotation,
        Tuple,
        default_collate,
        mo,
        nib,
        np,
        random,
        torch,
    )


@app.cell
def _(Any, Dict, Rotation, Tuple, nib, np, random):
    class NiftiPatchSampler:
        """
        Holds a single NIfTI scan and performs CPU-side logic:
        1. Loads scan data.
        2. Generates random transform parameters.
        3. Extracts and resamples source blocks (prisms) on the CPU.
        """
        def __init__(self, path_to_scan: str):
            self.nii_image = nib.as_closest_canonical(nib.load(path_to_scan))
            self.nii_data = self.nii_image.get_fdata()
            self.affine = np.array(self.nii_image.affine)
            self.voxel_spacing = np.array(self.nii_image.header.get_zooms()[:3])

            self.med = np.median(self.nii_data)
            self.std = np.std(self.nii_data)

            self.outlier_axis = self._get_outlier_axis()

        def _get_outlier_axis(
            self, similarity_threshold: float = 1.1
        ) -> Tuple[int, int, int]:
            """
            Determines patch shape in mm, setting the outlier axis to 1mm.
            (Logic adapted from your original)
            """
            shape = np.array(self.nii_data.shape)
            if max(shape) / min(shape) < similarity_threshold:
                outlier_axis_idx = random.randint(0, 2)
            else:
                diffs = [
                    abs(shape[0] - (shape[1] + shape[2]) / 2.0),
                    abs(shape[1] - (shape[0] + shape[2]) / 2.0),
                    abs(shape[2] - (shape[0] + shape[1]) / 2.0)
                ]
                outlier_axis_idx = np.argmax(diffs)

            return outlier_axis_idx

        def get_sample_parameters(self, 
                                  wc: float = None,
                                  ww: float = None,
                                  sampling_radius_mm: float = None,
                                  rotation_degrees: np.ndarray = None,
                                  base_patch_size: int = None,
                                  subset_center_vox: np.ndarray = None) -> Dict[str, Any]:
            """
            Generates all random parameters needed for one sample (one set of patches).
            This is called once per __getitem__.
            """
            results = {}

            # 1. Get windowing
            if wc is None:
                wc = random.uniform(self.med-self.std, self.med+self.std)
            if ww is None:
                ww = random.uniform(2*self.std, 6*self.std)
            results['wc'], results['ww'] = wc, ww
            results['w_min'], results['w_max'] = wc - 0.5 * ww, wc + 0.5 * ww

            # 2. Get sampling radius
            if sampling_radius_mm is None:
                sampling_radius_mm = random.uniform(20.0, 30.0)
            results['sampling_radius_mm'] = sampling_radius_mm

            # 3. Get rotation
            if rotation_degrees is None:
                rotation_degrees = np.array([
                    random.randint(-20, 20), 
                    random.randint(-20, 20), 
                    random.randint(-20, 20)
                ])
            results['rotation_degrees'] = rotation_degrees

            # 4. Get patch size
            if base_patch_size is None:
                base_patch_size = random.randint(16, 48)
            patch_shape_mm = [base_patch_size] * 3
            patch_shape_mm[self.outlier_axis] = 1
            results['patch_shape_mm'] = patch_shape_mm

            # 5. Get main center point
            if subset_center_vox is None:
                subset_center_vox = self._get_random_center_idx(sampling_radius_mm, patch_shape_mm)
            results['subset_center_vox'] = subset_center_vox

            # 6. Pre-calculate transform parameters
            results['transform_params'] = self._calculate_extraction_parameters(
                patch_shape_mm=patch_shape_mm,
                rotation_xyz_degrees=rotation_degrees,
                voxel_spacing=self.voxel_spacing
            )

            return results

        # TODO: EXPLAIN
        def get_source_data(
            self,
            n_patches: int,
            subset_center_vox: np.ndarray,
            sampling_radius_mm: float,
            transform_params: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Uses parameters to sample patch centers and extract/resample
            the source blocks (prisms) from the NIfTI data.
            """

            # 1. Sample patch center locations
            patch_centers_vox = self._sample_patch_centers(
                center_point_vox=subset_center_vox,
                sampling_radius_mm=sampling_radius_mm,
                num_patches=n_patches,
                voxel_spacing=self.voxel_spacing,
                volume_shape=self.nii_data.shape
            )

            # 2. Extract and resample all source blocks (CPU)
            source_block_shape_orig_vox = transform_params['source_block_shape_orig_vox']

            all_raw_blocks = [
                self._extract_single_block_cpu(
                    center_orig_vox=center_vox,
                    source_block_shape_orig_vox=source_block_shape_orig_vox
                ) for center_vox in patch_centers_vox
            ]

            # Stack into a (N, LR, PA, IS) array and add channel dim
            source_blocks = np.stack(all_raw_blocks).astype(np.float32)
            source_blocks = np.expand_dims(source_blocks, axis=1) # (N, 1, LR, PA, IS)

            # 3. Get positional information
            main_center_patient = self.convert_voxel_to_patient(subset_center_vox, self.affine)
            patch_centers_patient = self.convert_voxel_to_patient(patch_centers_vox, self.affine)

            relative_rotated_centers = self.calculate_rotated_relative_positions(
                 patch_centers_patient=patch_centers_patient,
                 main_center_patient=main_center_patient,
                 forward_rotation_matrix=transform_params["inverse_rotation_matrix"]
            )

            return {
                'source_blocks': source_blocks, # (N, 1, LR, PA, IS)
                'patch_centers_vox': patch_centers_vox,
                'relative_patch_centers_pt': relative_rotated_centers, # (N, 3)
                'prism_center_pt': main_center_patient # (3,)
            }

        def _extract_single_block_cpu(
            self,
            center_orig_vox: np.ndarray,
            source_block_shape_orig_vox: np.ndarray
        ) -> np.ndarray:
            """
            Performs the CPU-only part: safe extraction.
            Prefills with zeros when out of bounds
            """
            # --- 1. Safe Extraction from original volume ---
            starts = center_orig_vox - source_block_shape_orig_vox // 2
            ends = starts + source_block_shape_orig_vox
            source_block = np.zeros(source_block_shape_orig_vox, dtype=self.nii_data.dtype)

            crop_starts = np.maximum(starts, 0)
            crop_ends = np.minimum(ends, self.nii_data.shape)
            paste_starts = crop_starts - starts
            paste_ends = paste_starts + (crop_ends - crop_starts)

            source_block[tuple(slice(s, e) for s, e in zip(paste_starts, paste_ends))] = \
                self.nii_data[tuple(slice(s, e) for s, e in zip(crop_starts, crop_ends))]

            return source_block

        # TODO: This isn't really ideal tbh, could still easily lead to samples that go outside following rotation. 
        def _get_random_center_idx(self, sampling_radius_mm, patch_shape_mm):
            patch_shape_vox = np.ceil(np.array(patch_shape_mm) / self.voxel_spacing)
            patch_buffer_vox = np.ceil(patch_shape_vox * 1.5 / 2.0).astype(int)
            sampling_radius_vox = np.ceil(sampling_radius_mm / self.voxel_spacing).astype(int)
            total_buffer_vox = patch_buffer_vox + sampling_radius_vox
            min_idx = total_buffer_vox
            max_idx = np.array(self.nii_data.shape) - total_buffer_vox - 1
            if np.any(max_idx < min_idx):
                raise ValueError("Sampling radius/patch shape too large for image.")
            return np.array([np.random.randint(low, high + 1) for low, high in zip(min_idx, max_idx)])

        @staticmethod
        def _sample_patch_centers(
            center_point_vox: np.ndarray, sampling_radius_mm: float, num_patches: int,
            voxel_spacing: np.ndarray, volume_shape: Tuple[int, int, int]
        ) -> np.ndarray:
            center_point_mm = center_point_vox * voxel_spacing
            valid_centers_vox = []
            while len(valid_centers_vox) < num_patches:
                needed = num_patches - len(valid_centers_vox)
                points_to_sample = int(needed * 2.5) + 20

                sampled_relative_points_mm = np.random.uniform(
                    low=-sampling_radius_mm, high=sampling_radius_mm, size=(points_to_sample, 3))
                distances_sq = np.sum((sampled_relative_points_mm)**2, axis=1)
                in_sphere_mask = distances_sq <= sampling_radius_mm**2
                sampled_relative_points_mm = sampled_relative_points_mm[in_sphere_mask]
                sampled_relative_points_vox = np.round(sampled_relative_points_mm / voxel_spacing).astype(int)
                sampled_points_vox = center_point_vox + sampled_relative_points_vox
                in_volume_mask = np.all(sampled_points_vox >= 0, axis=1) & np.all(sampled_points_vox < volume_shape, axis=1)
                valid_centers_vox.extend(sampled_points_vox[in_volume_mask])
            return np.array(valid_centers_vox[:num_patches])

        @staticmethod
        def _calculate_extraction_parameters(
            patch_shape_mm: Tuple[int, int, int],
            rotation_xyz_degrees: Tuple[float, float, float],
            voxel_spacing: np.ndarray
        ) -> Dict[str, Any]:

            # --- 1. Final Patch Shape & Rotation ---
            # This is the desired shape *after* cropping.
            final_patch_shape_iso_vox = np.ceil(patch_shape_mm).astype(int)

            rotation = Rotation.from_euler('xyz', rotation_xyz_degrees, degrees=True)
            forward_rotation_matrix = rotation.as_matrix()
            # This matrix rotates from "rotated patch space" -> "original image space"
            inverse_rotation_matrix = np.linalg.inv(forward_rotation_matrix)

            # --- 2. Calculate TIGHT Bounding Box (The New Logic) ---

            # Get the 8 corners of the final patch, centered at (0,0,0)
            half_dims = final_patch_shape_iso_vox / 2.0
            corners = np.array([
                [-half_dims[0], -half_dims[1], -half_dims[2]],
                [-half_dims[0], -half_dims[1],  half_dims[2]],
                [-half_dims[0],  half_dims[1], -half_dims[2]],
                [-half_dims[0],  half_dims[1],  half_dims[2]],
                [ half_dims[0], -half_dims[1], -half_dims[2]],
                [ half_dims[0], -half_dims[1],  half_dims[2]],
                [ half_dims[0],  half_dims[1], -half_dims[2]],
                [ half_dims[0],  half_dims[1],  half_dims[2]]
            ])

            # Rotate the corners back into the original image's coordinate space
            rotated_corners = (inverse_rotation_matrix @ corners.T).T

            # Find the maximum extent of the rotated corners along each axis
            # This gives us the axis-aligned bounding box (AABB)
            aabb_half_dims = np.max(np.abs(rotated_corners), axis=0)

            # The full size of the source block is twice the half-dimensions
            source_block_shape_iso_vox = np.ceil(aabb_half_dims * 2.0) + 4

            # --- 3. Convert to Original Voxel Grid ---

            # We assume 1mm isotropic spacing for the GPU steps
            resampling_factor = voxel_spacing / np.array([1.0, 1.0, 1.0])

            # Calculate the shape to extract from the *original* NIfTI
            source_block_shape_orig_vox = np.ceil(
                source_block_shape_iso_vox / resampling_factor
            ).astype(int)

            # --- 4. Ensure odd dimensions (for stable center pixel) ---
            # Make sure the shapes are odd-numbered for a precise center
            source_block_shape_iso_vox = source_block_shape_iso_vox + (source_block_shape_iso_vox % 2 == 0)
            source_block_shape_orig_vox = source_block_shape_orig_vox + (source_block_shape_orig_vox % 2 == 0)

            return {
                "final_patch_shape_iso_vox": final_patch_shape_iso_vox.astype(int),
                "source_block_shape_iso_vox": source_block_shape_iso_vox.astype(int),
                "source_block_shape_orig_vox": source_block_shape_orig_vox.astype(int),
                "forward_rotation_matrix": forward_rotation_matrix,
                "inverse_rotation_matrix": inverse_rotation_matrix,
            }

        @staticmethod
        def convert_voxel_to_patient(points_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
            points_vox = np.atleast_2d(points_vox)
            homogeneous_coords = np.hstack((points_vox, np.ones((points_vox.shape[0], 1))))
            patient_coords = (affine @ homogeneous_coords.T).T[:, :3]
            return patient_coords.squeeze()

        @staticmethod
        def calculate_rotated_relative_positions(
            patch_centers_patient: np.ndarray, main_center_patient: np.ndarray,
            forward_rotation_matrix: np.ndarray
        ) -> np.ndarray:
            relative_vectors = patch_centers_patient - main_center_patient
            return (forward_rotation_matrix @ relative_vectors.T).T
    return (NiftiPatchSampler,)


@app.cell
def _(Any, Dataset, Dict, F, NiftiPatchSampler, Tuple, torch):
    class NiftiScanDataset(Dataset):
        """
        A PyTorch Dataset that generates samples using a NiftiPatchSampler.

        Each __getitem__ call returns one dictionary containing:
        - 'isotropic_blocks': (N, 1, D_iso, H_iso, W_iso) NumPy array
        - 'wc', 'ww': Floats
        - 'rotation_degrees': Tuple
        - 'transform_params': Dict
        - ... other metadata
        """
        def __init__(
            self,
            sampler: NiftiPatchSampler,
            n_patches_per_item: int,
            samples_per_epoch: int # Use a fixed number for random sampling
        ):
            self.sampler = sampler
            self.n_patches = n_patches_per_item
            self.samples_per_epoch = samples_per_epoch

        def __len__(self):
            return self.samples_per_epoch

        def __getitem__(self, idx):
            # 1. Get random parameters for this item
            param_dict = self.sampler.get_sample_parameters()

            # 2. Get CPU-extracted data using these params
            data_dict = self.sampler.get_source_data(
                n_patches=self.n_patches,
                subset_center_vox=param_dict['subset_center_vox'],
                sampling_radius_mm=param_dict['sampling_radius_mm'],
                transform_params=param_dict['transform_params']
            )

            # 3. Combine and return the single item
            # The collate_fn will batch these dictionaries
            return {**param_dict, **data_dict}


    class GPUPatchTransformer:
        """
        A callable class to apply the GPU part of the transform pipeline
        using pure PyTorch functions (F.affine_grid, F.grid_sample).
        """
        def __init__(self, device: torch.device, final_model_shape: Tuple[int, int, int] = (16, 16, 1)):
            self.device = device
            self.final_model_shape = final_model_shape

        @staticmethod
        def _normalize_batch(batch_tensor, wc, ww, out_min=-1.0, out_max=1.0):
            """Helper for batched windowing."""
            w_min = wc - 0.5 * ww
            w_max = wc + 0.5 * ww
            if w_max <= w_min: w_max = w_min + 1e-6

            clipped = torch.clamp(batch_tensor, w_min, w_max)
            scaled_01 = (clipped - w_min) / (w_max - w_min)
            return scaled_01 * (out_max - out_min) + out_min

        @staticmethod
        def _rotate_batch_pytorch(
            batch_tensor: torch.Tensor, # Shape (N, 1, D, H, W)
            rotation_xyz_degrees: torch.Tensor # Shape (3,)
        ) -> torch.Tensor:
            """
            Applies 3D rotation using F.affine_grid and F.grid_sample.
            This version is PURE TORCH and stays on the GPU.
            """
            device = batch_tensor.device
            n_patches = batch_tensor.shape[0]

            # 1. Convert degrees to radians ON-DEVICE
            # We unsqueeze to (3, 1) for broadcasting later
            angles_rad = torch.deg2rad(rotation_xyz_degrees)

            # Grid 'x' rotation (NIfTI Z-axis / Axial) comes from input[2]
            angle_x = angles_rad[2] 

            # Grid 'y' rotation (NIfTI Y-axis / Coronal) comes from input[1]
            angle_y = angles_rad[1]

            # Grid 'z' rotation (NIfTI X-axis / Sagittal) comes from input[0]
            angle_z = angles_rad[0]

            # 2. Create individual rotation matrices ON-DEVICE
            cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
            Rx = torch.tensor([
                [1, 0, 0], 
                [0, cos_x, -sin_x], 
                [0, sin_x, cos_x]
            ], dtype=torch.float32, device=device)

            cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
            Ry = torch.tensor([
                [cos_y, 0, sin_y], 
                [0, 1, 0], 
                [-sin_y, 0, cos_y]
            ], dtype=torch.float32, device=device)

            cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)
            Rz = torch.tensor([
                [cos_z, -sin_z, 0], 
                [sin_z, cos_z, 0], 
                [0, 0, 1]
            ], dtype=torch.float32, device=device)

            # 3. Combine rotations and create affine matrix ON-DEVICE
            R = Rz @ Ry @ Rx
            affine_matrix_3x4 = torch.eye(3, 4, dtype=torch.float32, device=device)
            affine_matrix_3x4[:3, :3] = R

            # 4. Expand to batch size (N, 3, 4)
            theta = affine_matrix_3x4.unsqueeze(0).expand(n_patches, -1, -1)

            # 5. Generate grid and sample (all on GPU)
            grid = F.affine_grid(theta, batch_tensor.shape, align_corners=False)
            rotated_batch = F.grid_sample(
                batch_tensor, 
                grid, 
                mode='bilinear',
                padding_mode='zeros', 
                align_corners=False
            )

            return rotated_batch


        def __call__(self, batch: Dict[str, Any]) -> torch.Tensor:
            """
            Applies resampling, rotation, cropping, and windowing to a collated batch.
            """
            final_patches_list = []

            # --- MODIFIED ---
            # Get batch size from the length of the list, not the tensor shape
            batch_size = len(batch['source_blocks'])

            for i in range(batch_size):
                # --- MODIFIED ---
                # Get the i-th tensor from the list
                blocks_n = batch['source_blocks'][i].to(self.device)

                # --- Get item-specific transform params ---
                params_i = batch['transform_params'][i]

                # (The rest of your function works perfectly)
                rot_degrees_i = batch['rotation_degrees'][i]
                wc_i = batch['wc'][i].item()
                ww_i = batch['ww'][i].item()
                final_shape_i = params_i['final_patch_shape_iso_vox']

                # --- 1: (GPU) Resample to Isotropic ---
                target_iso_shape = params_i['source_block_shape_iso_vox']
                target_iso_shape_ints = tuple(int(dim) for dim in target_iso_shape)

                # This F.interpolate now takes a variable-sized input...
                isotropic_blocks = F.interpolate(
                    blocks_n,
                    size=tuple(target_iso_shape_ints),
                    mode='trilinear',
                    align_corners=False
                )
                # ...and produces a fixed-size output, (N, 1, D_iso, H_iso, W_iso)

                # --- 2: (GPU) Apply Rotation ---
                rotated_blocks = self._rotate_batch_pytorch(isotropic_blocks, rot_degrees_i)

                # --- 3: (GPU) Apply Center Crop ---
                current_shape_dhw = rotated_blocks.shape[2:]
                starts = [(rs - fs) // 2 for rs, fs in zip(current_shape_dhw, final_shape_i)]
                ends = [s + fs for s, fs in zip(starts, final_shape_i)]
                cropped_blocks = rotated_blocks[
                    :, :,
                    starts[0]:ends[0],
                    starts[1]:ends[1],
                    starts[2]:ends[2]
                ]

                # --- 4: (GPU) Apply Windowing ---
                normalized_blocks = self._normalize_batch(cropped_blocks, wc_i, ww_i)

                # --- 5: (GPU) Resize to model input size ---
                final_patch = F.interpolate(
                    normalized_blocks,
                    size=self.final_model_shape,
                    mode='trilinear',
                    align_corners=False
                )

                final_patches_list.append(final_patch)

            # Stacks the list of (N, C, D, H, W) tensors into (B, N, C, D, H, W)
            return torch.stack(final_patches_list, dim=0)
    return (GPUPatchTransformer,)


@app.cell
def _(Any, Dict, List, default_collate, torch):
    def custom_collate_fn(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function that handles a list of dictionaries.
        """
        if not isinstance(batch_list[0], dict):
            return default_collate(batch_list)

        collated_batch = {}
        keys = batch_list[0].keys()

        for key in keys:
            values = [d[key] for d in batch_list]

            if key == 'transform_params':
                # For dicts, just return a list
                collated_batch[key] = values

            elif key == 'source_blocks':
                # For source blocks, convert each to tensor (with copy) 
                # and return as a list
                collated_batch[key] = [torch.tensor(v) for v in values]

            else:
                # For everything else (wc, ww, rotation_degrees, etc.), 
                # try to use default collate to stack them into tensors
                try:
                    collated_batch[key] = default_collate(values)
                except TypeError:
                    # Failsafe for un-coll-atable types, just list them
                    collated_batch[key] = values

        return collated_batch
    return


@app.cell(column=1)
def _(NiftiPatchSampler, mo):
    sampler = NiftiPatchSampler(
        path_to_scan="/cbica/home/gangarav/rsna23/nifti/10046.nii.gz"
    )

    mo.vstack([
        mo.md(f"nii_data shape: `{sampler.nii_data.shape}`"),
        mo.md(f"voxel_spacing: `{sampler.voxel_spacing}`"),
        mo.md(f"outlier_axis: `{sampler.outlier_axis}`")
    ])
    return (sampler,)


@app.cell
def _(mo):
    wc_field = mo.ui.number(value=0.0, label="Window Center (wc)", step=1.0)
    ww_field = mo.ui.number(value=400.0, label="Window Width (ww)", step=1.0)
    sampling_radius_mm_field = mo.ui.number(value=25.0, label="Sampling Radius (mm)", step=0.5)
    rotation_field = mo.ui.text(value="0, 0, 0", label="Rotation (degrees x,y,z)")
    base_patch_size_field = mo.ui.number(value=32, label="Base Patch Size", start=16, stop=129, step=1)
    center_point_vox = mo.ui.text(value="256, 256, 300", label="Center Point Vox")

    mo.vstack([
        wc_field,
        ww_field,
        sampling_radius_mm_field,
        rotation_field,
        base_patch_size_field,
        mo.md("**Subset Center Voxel**"),
        center_point_vox
    ])
    return (
        base_patch_size_field,
        center_point_vox,
        rotation_field,
        sampling_radius_mm_field,
        wc_field,
        ww_field,
    )


@app.cell
def _(
    base_patch_size_field,
    center_point_vox,
    np,
    rotation_field,
    sampler,
    sampling_radius_mm_field,
    wc_field,
    ww_field,
):
    param_dict = sampler.get_sample_parameters(
        wc=float(wc_field.value),
        ww=float(ww_field.value),
        sampling_radius_mm=float(sampling_radius_mm_field.value),
        rotation_degrees=np.array([float(x.strip()) for x in rotation_field.value.split(',')]),
        base_patch_size=int(base_patch_size_field.value),
        subset_center_vox=np.array([int(x.strip()) for x in center_point_vox.value.split(',')])
    )
    return (param_dict,)


@app.cell
def _(mo, param_dict, sampler):
    patch_centers_vox = sampler._sample_patch_centers(
        center_point_vox=param_dict['subset_center_vox'],
        sampling_radius_mm=param_dict['sampling_radius_mm'],
        num_patches=10,
        voxel_spacing=sampler.voxel_spacing,
        volume_shape=sampler.nii_data.shape
    )

    patch_center_field = mo.ui.text(value=', '.join(map(str, patch_centers_vox[0])), label="First Patch Center")
    patch_center_field
    return patch_center_field, patch_centers_vox


@app.cell
def _(mo, np, param_dict, patch_center_field, patch_centers_vox, sampler):
    source_block_shape_orig_vox = param_dict['transform_params']['source_block_shape_orig_vox']

    updated_patch_centers_vox = patch_centers_vox[:]
    updated_patch_centers_vox[0] = np.array([int(x.strip()) for x in patch_center_field.value.split(',')])

    # 3. Get positional information
    main_center_patient = sampler.convert_voxel_to_patient(param_dict['subset_center_vox'], sampler.affine)
    patch_centers_patient = sampler.convert_voxel_to_patient(updated_patch_centers_vox, sampler.affine)

    relative_rotated_centers = sampler.calculate_rotated_relative_positions(
         patch_centers_patient=patch_centers_patient,
         main_center_patient=main_center_patient,
         forward_rotation_matrix=param_dict['transform_params']["inverse_rotation_matrix"]
    )

    # Format arrays to 3 significant figures for display
    main_center_labeled = np.array2string(main_center_patient, precision=3, separator=', ', suppress_small=True)
    patch_centers_labeled = np.array2string(patch_centers_patient[0], precision=3, separator=', ', suppress_small=True)
    relative_rotated_labeled = np.array2string(relative_rotated_centers[0], precision=3, separator=', ', suppress_small=True)

    mo.md(f"""
    **Main Center (Patient Coordinates):** `{main_center_labeled}`  
    **Patch Centers (Patient Coordinates):** `{patch_centers_labeled}`  
    **Relative Rotated Centers:** `{relative_rotated_labeled}`
    """)
    return source_block_shape_orig_vox, updated_patch_centers_vox


@app.cell
def _(np, patch_centers_vox, sampler, source_block_shape_orig_vox):
    all_raw_blocks = [
        sampler._extract_single_block_cpu(
            center_orig_vox=center_vox,
            source_block_shape_orig_vox=source_block_shape_orig_vox
        ) for center_vox in patch_centers_vox
    ]

    # Stack into a (N, LR, PA, IS) array and add channel dim
    source_blocks = np.stack(all_raw_blocks).astype(np.float32)
    source_blocks = np.expand_dims(source_blocks, axis=1) # (N, 1, LR, PA, IS)
    return (source_blocks,)


@app.cell
def _(mo, source_blocks):
    patch_selector = mo.ui.slider(
        start=0,
        stop=source_blocks.shape[0] - 1,
        value=0,
        label="Select Patch Index"
    )
    patch_selector
    return (patch_selector,)


@app.cell
def _(mo, patch_selector, source_blocks):
    mo.hstack([mo.image(src=source_blocks[patch_selector.value,0,:,:,source_blocks.shape[-1]//2], height=64), source_blocks.shape])
    return


@app.cell
def _(DEVICE, GPUPatchTransformer):
    t = GPUPatchTransformer(
        device=DEVICE,
        final_model_shape=(128, 128, 1)
    )
    return (t,)


@app.cell
def _(DEVICE, mo, param_dict, patch_selector, source_blocks, t, torch):
    patches = t({
        'source_blocks': [torch.tensor(source_blocks).to(DEVICE)],
        'transform_params': [param_dict['transform_params']],
        'rotation_degrees': torch.tensor(param_dict['rotation_degrees']).to(DEVICE).unsqueeze(0),
        'wc': torch.tensor([param_dict['wc']]).to(DEVICE),
        'ww': torch.tensor([param_dict['ww']]).to(DEVICE)
    })

    mo.hstack([mo.image(src=patches[0,patch_selector.value,0,:,:,0].cpu().numpy(), height=64), patches.shape])
    return


@app.cell
def _(mo, patch_selector, sampler, updated_patch_centers_vox):
    default_slice = sampler.nii_data.shape[sampler.outlier_axis] // 2
    if patch_selector.value < len(updated_patch_centers_vox):
        patch_center = updated_patch_centers_vox[patch_selector.value]
        default_slice = int(patch_center[sampler.outlier_axis])

    slider = mo.ui.slider(
        start=0,
        stop=sampler.nii_data.shape[sampler.outlier_axis] - 1,
        value=default_slice,
        label="Scan Slice Viewer"
    )
    slider
    return (slider,)


@app.cell
def _(mo, sampler, slider):
    slicer = [slice(None)] * 3
    slicer[sampler.outlier_axis] = slider.value
    slice_2d = sampler.nii_data[tuple(slicer)]

    mo.image(src=slice_2d)
    return


@app.cell
def _():
    # # Get rotation matrix from parameters
    # rotation_degrees = param_dict['rotation_degrees']
    # rotation = Rotation.from_euler('xyz', rotation_degrees, degrees=True)
    # rotation_matrix = rotation.as_matrix()

    # # Get the full scan data
    # scan_data = sampler.nii_data.copy()

    # # Create a grid of indices for the scan
    # shape = scan_data.shape
    # indices = np.indices(shape).reshape(3, -1).T

    # # Convert to physical coordinates using affine
    # ones = np.ones((indices.shape[0], 1))
    # homogeneous_coords = np.hstack((indices, ones))
    # physical_coords = (sampler.affine @ homogeneous_coords.T).T[:, :3]

    # # Center the physical coordinates around the main center
    # _main_center_patient = sampler.convert_voxel_to_patient(param_dict['subset_center_vox'], sampler.affine)
    # centered_coords = physical_coords - _main_center_patient

    # # Apply inverse rotation to get rotated coordinates
    # inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    # rotated_coords = (inverse_rotation_matrix @ centered_coords.T).T

    # # Add back the main center
    # rotated_physical_coords = rotated_coords + _main_center_patient

    # # Convert back to voxel coordinates
    # affine_inv = np.linalg.inv(sampler.affine)
    # homogeneous_rotated = np.hstack((rotated_physical_coords, np.ones((rotated_physical_coords.shape[0], 1))))
    # voxel_coords = (affine_inv @ homogeneous_rotated.T).T[:, :3]

    # # Reshape back to 3D grid
    # voxel_coords = voxel_coords.reshape(shape[0], shape[1], shape[2], 3)

    # # Use ndimage.map_coordinates for interpolation
    # rotated_scan = ndimage.map_coordinates(scan_data, 
    #                                         [voxel_coords[:, :, :, 0],
    #                                          voxel_coords[:, :, :, 1], 
    #                                          voxel_coords[:, :, :, 2]],
    #                                         order=1, 
    #                                         mode='constant', 
    #                                         cval=sampler.med)

    # # Display the rotated scan
    # slicer_rot = [slice(None)] * 3
    # slicer_rot[sampler.outlier_axis] = slider.value
    # rotated_slice = rotated_scan[tuple(slicer_rot)]

    # mo.image(src=rotated_slice)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
