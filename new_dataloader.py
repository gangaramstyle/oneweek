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
    import torchvision.transforms as transforms
    from scipy import ndimage
    from typing import Tuple, Dict, Any, Optional, List
    import marimo as mo
    import pandas as pd
    import time
    import math
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
def _(Any, Dict, Rotation, nib, np, random):
    class NiftiPatchSampler:
        def __init__(self, path_to_scan: str):
            self.nii_image = nib.as_closest_canonical(nib.load(path_to_scan))
            self.nii_data = self.nii_image.get_fdata()
            self.affine = np.array(self.nii_image.affine)
            self.voxel_spacing = np.array(self.nii_image.header.get_zooms()[:3])
            self.med = np.median(self.nii_data)
            self.std = np.std(self.nii_data)
            self.outlier_axis = self._get_outlier_axis()

        def _get_outlier_axis(self, similarity_threshold: float = 1.1) -> int:
            shape = np.array(self.nii_data.shape)
            if max(shape) / min(shape) < similarity_threshold:
                return random.randint(0, 2)
            diffs = [
                abs(shape[0] - (shape[1] + shape[2]) / 2.0),
                abs(shape[1] - (shape[0] + shape[2]) / 2.0),
                abs(shape[2] - (shape[0] + shape[1]) / 2.0)
            ]
            return np.argmax(diffs)

        def get_sample_parameters(self, 
                                  wc: float = None,
                                  ww: float = None,
                                  sampling_radius_mm: float = None,
                                  rotation_quat: np.ndarray = None, # Changed from degrees
                                  base_patch_size: int = None,
                                  subset_center_vox: np.ndarray = None) -> Dict[str, Any]:
            results = {}

            # 1. Windowing
            if wc is None: wc = random.uniform(self.med-self.std, self.med+self.std)
            if ww is None: ww = random.uniform(2*self.std, 6*self.std)
            results['wc'], results['ww'] = wc, ww

            # 2. Sampling Radius
            if sampling_radius_mm is None:
                sampling_radius_mm = random.uniform(20.0, 30.0)
            results['sampling_radius_mm'] = sampling_radius_mm

            # 3. Rotation (Generate Euler first for control, then convert to Quat)
            if rotation_quat is None:
                euler_deg = np.array([random.randint(-20, 20) for _ in range(3)])
                # Scipy uses (x, y, z, w)
                rotation_quat = Rotation.from_euler('xyz', euler_deg, degrees=True).as_quat()

            results['rotation_quat'] = rotation_quat

            # 4. Patch Size
            if base_patch_size is None:
                base_patch_size = random.randint(16, 48)
            patch_shape_mm = [base_patch_size] * 3
            patch_shape_mm[self.outlier_axis] = 1
            results['patch_shape_mm'] = patch_shape_mm

            # 5. Center Point
            if subset_center_vox is None:
                subset_center_vox = self._get_random_center_idx(sampling_radius_mm, patch_shape_mm)
            results['subset_center_vox'] = subset_center_vox

            # 6. Pre-calculate transform params (Using Quat)
            results['transform_params'] = self._calculate_extraction_parameters(
                patch_shape_mm=patch_shape_mm,
                rotation_quat=rotation_quat,
                voxel_spacing=self.voxel_spacing
            )

            return results

        def get_source_data(self, n_patches, subset_center_vox, sampling_radius_mm, transform_params):
            # 1. Sample centers
            patch_centers_vox = self._sample_patch_centers(
                center_point_vox=subset_center_vox,
                sampling_radius_mm=sampling_radius_mm,
                num_patches=n_patches,
                voxel_spacing=self.voxel_spacing,
                volume_shape=self.nii_data.shape
            )

            # 2. Extract blocks
            source_block_shape = transform_params['source_block_shape_orig_vox']
            all_raw_blocks = [
                self._extract_single_block_cpu(c, source_block_shape) 
                for c in patch_centers_vox
            ]

            source_blocks = np.stack(all_raw_blocks).astype(np.float32)
            source_blocks = np.expand_dims(source_blocks, axis=1)

            # 3. Positional Info
            main_center_pt = self.convert_voxel_to_patient(subset_center_vox, self.affine)
            patch_centers_pt = self.convert_voxel_to_patient(patch_centers_vox, self.affine)

            relative_rotated = self.calculate_rotated_relative_positions(
                 patch_centers_patient=patch_centers_pt,
                 main_center_patient=main_center_pt,
                 forward_rotation_matrix=transform_params["inverse_rotation_matrix"]
            )

            return {
                'source_blocks': source_blocks,
                'patch_centers_vox': patch_centers_vox,
                'relative_patch_centers_pt': relative_rotated,
                'prism_center_pt': main_center_pt
            }

        def _extract_single_block_cpu(self, center_orig_vox, source_block_shape_orig_vox):
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

        def _get_random_center_idx(self, sampling_radius_mm, patch_shape_mm):
            patch_shape_vox = np.ceil(np.array(patch_shape_mm) / self.voxel_spacing)
            # Safety buffer
            buffer_vox = (np.ceil(patch_shape_vox * 1.5 / 2.0) + np.ceil(sampling_radius_mm / self.voxel_spacing)).astype(int)
            min_idx = buffer_vox
            max_idx = np.array(self.nii_data.shape) - buffer_vox - 1
            return np.array([np.random.randint(l, h + 1) for l, h in zip(min_idx, max_idx)])

        @staticmethod
        def _sample_patch_centers(center_point_vox, sampling_radius_mm, num_patches, voxel_spacing, volume_shape):
            valid_centers = []
            while len(valid_centers) < num_patches:
                needed = num_patches - len(valid_centers)
                # Sample relative points in mm
                pts_mm = np.random.uniform(-sampling_radius_mm, sampling_radius_mm, size=(needed*3 + 20, 3))
                pts_mm = pts_mm[np.sum(pts_mm**2, axis=1) <= sampling_radius_mm**2]

                pts_vox = np.round(pts_mm / voxel_spacing).astype(int) + center_point_vox
                in_vol = np.all((pts_vox >= 0) & (pts_vox < volume_shape), axis=1)
                valid_centers.extend(pts_vox[in_vol])
            return np.array(valid_centers[:num_patches])

        @staticmethod
        def _calculate_extraction_parameters(patch_shape_mm, rotation_quat, voxel_spacing):
            final_patch_shape_iso = np.ceil(patch_shape_mm).astype(int)

            # --- CHANGED: Use Quaternion ---
            # Scipy rotation from quaternion (x, y, z, w)
            rotation = Rotation.from_quat(rotation_quat)
            forward_rotation_matrix = rotation.as_matrix()
            inverse_rotation_matrix = np.linalg.inv(forward_rotation_matrix)

            # Calculate AABB
            half_dims = final_patch_shape_iso / 2.0
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

            rotated_corners = (inverse_rotation_matrix @ corners.T).T
            aabb_half_dims = np.max(np.abs(rotated_corners), axis=0)
            source_block_shape_iso = np.ceil(aabb_half_dims * 2.0)

            # Convert to Original Grid
            resampling_factor = voxel_spacing / np.array([1.0, 1.0, 1.0])
            source_block_shape_orig = np.ceil(source_block_shape_iso / resampling_factor).astype(int)

            # Ensure odd
            source_block_shape_iso += (source_block_shape_iso % 2 == 0)
            source_block_shape_orig += (source_block_shape_orig % 2 == 0)

            return {
                "final_patch_shape_iso_vox": final_patch_shape_iso,
                "source_block_shape_iso_vox": source_block_shape_iso.astype(int),
                "source_block_shape_orig_vox": source_block_shape_orig.astype(int),
                "forward_rotation_matrix": forward_rotation_matrix,
                "inverse_rotation_matrix": inverse_rotation_matrix,
            }

        @staticmethod
        def convert_voxel_to_patient(points_vox, affine):
            points_vox = np.atleast_2d(points_vox)
            homo = np.hstack((points_vox, np.ones((points_vox.shape[0], 1))))
            return (affine @ homo.T).T[:, :3].squeeze()

        @staticmethod
        def calculate_rotated_relative_positions(patch_centers_patient, main_center_patient, forward_rotation_matrix):
            vecs = patch_centers_patient - main_center_patient
            return (forward_rotation_matrix @ vecs.T).T
    return (NiftiPatchSampler,)


@app.cell
def _(Any, Dataset, Dict, F, Tuple, torch):
    class NiftiScanDataset(Dataset):
        def __init__(self, sampler, n_patches_per_item, samples_per_epoch):
            self.sampler = sampler
            self.n_patches = n_patches_per_item
            self.samples_per_epoch = samples_per_epoch

        def __len__(self): return self.samples_per_epoch

        def __getitem__(self, idx):
            param_dict = self.sampler.get_sample_parameters()
            data_dict = self.sampler.get_source_data(
                n_patches=self.n_patches,
                subset_center_vox=param_dict['subset_center_vox'],
                sampling_radius_mm=param_dict['sampling_radius_mm'],
                transform_params=param_dict['transform_params']
            )
            return {**param_dict, **data_dict}

    class GPUPatchTransformer:
        def __init__(self, device: torch.device, final_model_shape: Tuple[int, int, int] = (16, 16, 1)):
            self.device = device
            self.final_model_shape = final_model_shape

        @staticmethod
        def _normalize_batch(batch_tensor, wc, ww, out_min=-1.0, out_max=1.0):
            w_min, w_max = wc - 0.5 * ww, wc + 0.5 * ww
            if w_max <= w_min: w_max = w_min + 1e-6
            clipped = torch.clamp(batch_tensor, w_min, w_max)
            return ((clipped - w_min) / (w_max - w_min)) * (out_max - out_min) + out_min

        @staticmethod
        def _rotate_batch_pytorch(batch_tensor, rotation_quats, final_shape):
            """
            Applies 3D rotation using Quaternions (x, y, z, w).
            """
            device = batch_tensor.device
            n_patches = batch_tensor.shape[0]

            # --- FIX 1: Conjugate the Quaternion ---
            # The CPU code uses the INVERSE matrix (Target -> Source).
            # The inverse of a unit quaternion is its conjugate: (-x, -y, -z, w)
            q = F.normalize(rotation_quats, p=2, dim=1)
            x, y, z, w = -q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]

            # Construct Rotation Matrix elements
            tx = 2.0 * x
            ty = 2.0 * y
            tz = 2.0 * z
            twx, twy, twz = tx * w, ty * w, tz * w
            txx, txy, txz = tx * x, ty * x, tz * x
            tyy, tyz, tzz = ty * y, tz * y, tz * z

            R = torch.zeros((n_patches, 3, 3), device=device, dtype=torch.float32)

            # Row 0
            R[:, 0, 0] = 1.0 - (tyy + tzz)
            R[:, 0, 1] = txy - twz
            R[:, 0, 2] = txz + twy

            # Row 1
            R[:, 1, 0] = txy + twz
            R[:, 1, 1] = 1.0 - (txx + tzz)
            R[:, 1, 2] = tyz - twx

            # Row 2
            R[:, 2, 0] = txz - twy
            R[:, 2, 1] = tyz + twx
            R[:, 2, 2] = 1.0 - (txx + tyy)

            theta = torch.zeros((n_patches, 3, 4), device=device, dtype=torch.float32)
            theta[:, :3, :3] = R

            target_grid_shape = (n_patches, 1) + final_shape 
            grid = F.affine_grid(theta, target_grid_shape, align_corners=False)
            grid_max_per_axis = grid.amax(dim=(0,1,2,3))
            print("grid_max_per_axis", grid_max_per_axis)

            # Grid Sample
            rotated_batch = F.grid_sample(
                batch_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False
            )
            return rotated_batch

        def __call__(self, batch: Dict[str, Any]) -> torch.Tensor:
            final_patches_list = []
            batch_size = len(batch['source_blocks'])

            for i in range(batch_size):
                blocks_n = batch['source_blocks'][i].to(self.device)
                params_i = batch['transform_params'][i]

                rot_quat_i = batch['rotation_quat'][i]
                if not isinstance(rot_quat_i, torch.Tensor):
                    rot_quat_i = torch.tensor(rot_quat_i, dtype=torch.float32)
                rot_quat_i = rot_quat_i.to(self.device).view(-1, 4) 

                wc_i = batch['wc'][i].item()
                ww_i = batch['ww'][i].item()
                final_shape_i = tuple(int(dim) for dim in params_i['final_patch_shape_iso_vox'])
                target_iso_shape = tuple(int(dim) for dim in params_i['source_block_shape_iso_vox'])

                # 1. Resample to Isotropic
                # NOTE: We do NOT transpose here yet because F.interpolate operates on the "image" axes 
                # regardless of meaning. We just need it to be isotropic.
                isotropic_blocks = F.interpolate(
                    blocks_n, size=target_iso_shape, mode='trilinear', align_corners=False
                )

                # --- FIX 2: Transpose Axes for Grid Sample ---
                # Current Shape: (N, 1, X, Y, Z)
                # PyTorch Expects: (N, 1, Depth, Height, Width) -> Width is sampled by x
                # We want "Width" to be "X". So we must put X at the end.
                # New Shape: (N, 1, Z, Y, X)
                isotropic_blocks_permuted = isotropic_blocks.permute(0, 1, 4, 3, 2)

                # 2. Rotate using Quaternions
                # We pass the permuted block. The grid generator logic (x,y,z) now correctly
                # maps to (X, Y, Z) of the data.
                cropped_blocks_permuted = self._rotate_batch_pytorch(
                    isotropic_blocks_permuted, rot_quat_i, (final_shape_i[2], final_shape_i[1], final_shape_i[0])
                )

                # Transpose BACK to (X, Y, Z) standard for medical imaging
                cropped_blocks = cropped_blocks_permuted.permute(0, 1, 4, 3, 2)

                # 3. Window & Resize
                normalized_blocks = self._normalize_batch(cropped_blocks, wc_i, ww_i)
                final_patch = F.interpolate(
                    normalized_blocks, size=self.final_model_shape, mode='trilinear', align_corners=False
                )
                final_patches_list.append(final_patch)

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

    # --- CHANGED: Label and default value for Quaternion ---
    rotation_field = mo.ui.text(value="0, 0, 0", label="Rotation  (x,y,z)")

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
    Rotation,
    base_patch_size_field,
    center_point_vox,
    np,
    rotation_field,
    sampler,
    sampling_radius_mm_field,
    wc_field,
    ww_field,
):
    # 1. Parse the degrees from UI
    degrees_xyz = [float(x.strip()) for x in rotation_field.value.split(',')]

    # 2. Convert to Quaternion (x, y, z, w)
    # 'xyz' is the standard Euler order; change if you need 'zyx'
    user_quat = Rotation.from_euler('xyz', degrees_xyz, degrees=True).as_quat()
    print(user_quat)

    # 3. Pass the calculated quaternion to the sampler
    param_dict = sampler.get_sample_parameters(
        wc=float(wc_field.value),
        ww=float(ww_field.value),
        sampling_radius_mm=float(sampling_radius_mm_field.value),
        rotation_quat=user_quat, # Pass the converted quaternion here
        base_patch_size=int(base_patch_size_field.value),
        subset_center_vox=np.array([int(x.strip()) for x in center_point_vox.value.split(',')])
    )
    param_dict
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
    **Source Block Shape:** `{source_block_shape_orig_vox}`
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
    source_blocks.shape
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
    rot_quat = torch.tensor(param_dict['rotation_quat']).to(DEVICE).unsqueeze(0) # Shape (1, 4)

    patches = t({
        'source_blocks': [torch.tensor(source_blocks).to(DEVICE)],
        'transform_params': [param_dict['transform_params']],
        'rotation_quat': rot_quat, # Changed key
        'wc': torch.tensor([param_dict['wc']]).to(DEVICE),
        'ww': torch.tensor([param_dict['ww']]).to(DEVICE)
    })

    mo.hstack([mo.image(src=patches[0,patch_selector.value,0,:,:,patches.shape[-1]//2].cpu().numpy(), height=64), patches.shape])
    return (patches,)


@app.cell
def _(patches):
    patches.shape
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
