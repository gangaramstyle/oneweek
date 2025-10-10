import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")

with app.setup:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info
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
    from mim_lightning_train import RadiographyEncoder
    from data_loader import nifti_scan


@app.class_definition
class ValidationDataset(IterableDataset):
    def __init__(self, metadata, patch_shape=None, n_patches=None, sampling_radius_mm=10, n_samples=1, n_randoms=1):
        super().__init__()
        self.metadata = metadata.sample(frac=1).reset_index(drop=True) # Ensure simple integer index
        self.patch_shape = patch_shape
        self.n_patches = n_patches
        self.n_samples = n_samples
        self.n_randoms = n_randoms
        self.sampling_radius_mm = sampling_radius_mm

        # Define labels
        self.sample_labels = [1, 2, 3, 4, 5, 6, 7, 10]
        self.random_labels = [0]
        self.all_labels = self.sample_labels + self.random_labels

    def generate_training_sample(self, scan, n_patches, sampling_radius_mm, subset_center=None):
        # Assuming this function is defined as before
        sample = scan.train_sample(n_patches, subset_center=subset_center, sampling_radius_mm=sampling_radius_mm)
        patches = torch.from_numpy(sample["normalized_patches"]).to(torch.float32)
        patch_coords = torch.from_numpy(sample['relative_patch_centers_pt']).to(torch.float32)
        return patches, sample["patch_centers_vox"], patch_coords, sample["subset_center"]

    def __iter__(self):
        # --- OPTIMIZATION: Handle multi-worker data splitting ---
        worker_info = get_worker_info()
        if worker_info is None:  # Single-process loading
            start, end = 0, len(self.metadata)
        else: # Multi-process loading
            per_worker = int(np.ceil(len(self.metadata) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.metadata))

        # Iterate over this worker's assigned slice of the metadata
        for idx in range(start, end):
            row = self.metadata.iloc[idx]
            try:
                img_path = row["raw_path"].replace("../../", "../")
                seg_path = img_path.replace('imagesTr', 'cow_seg_labelsTr').replace('_0000', '')

                scan = nifti_scan(path_to_scan=img_path, median=row["median"], stdev=row["stdev"], base_patch_size=self.patch_shape)
                seg = nib.as_closest_canonical(nib.load(seg_path)).get_fdata()

            except Exception as e:
                # Optional: print(f"Skipping {img_path} due to {e}")
                continue

            # --- OPTIMIZATION: Pre-calculate all label coordinates once per scan ---
            locations_by_label = {
                label: np.argwhere(seg == float(label)) for label in self.all_labels
            }

            # --- Sample from positive labels ---
            for label in self.sample_labels:
                possible_centers = locations_by_label.get(label, [])
                if len(possible_centers) == 0:
                    continue

                for _ in range(self.n_samples):
                    subset_center = random.choice(possible_centers)
                    patches, patch_coords_vox, patch_coords, center = self.generate_training_sample(scan, self.n_patches, self.sampling_radius_mm, subset_center=subset_center)
                    yield patches, patch_coords_vox, patch_coords, label, img_path, center

            # --- Sample from random/background labels ---
            for label in self.random_labels:
                possible_centers = locations_by_label.get(label, [])
                if len(possible_centers) == 0:
                    continue

                for _ in range(self.n_randoms):
                    subset_center = random.choice(possible_centers)
                    patches, patch_coords_vox, patch_coords, center = self.generate_training_sample(scan, self.n_patches, self.sampling_radius_mm, subset_center=subset_center)
                    yield patches, patch_coords_vox, patch_coords, label, img_path, center


@app.cell(hide_code=True)
def _():
    base_patch_shape = mo.ui.text(value="16", label="Patch Size (mm):")
    n_patches_input = mo.ui.text(value="64", label="Num Patches:")

    mo.vstack([base_patch_shape, n_patches_input])
    return base_patch_shape, n_patches_input


@app.cell
def _():
    sampling_radius_mm_input = mo.ui.text(value="", label="Sampling Radius (mm) - leave blank for random:")
    rotation_input = mo.ui.text(value="", label="Rotation (x,y,z degrees) - leave blank for random:")
    run_id_input = mo.ui.text(
        value="", # A default value to start with
        label="Enter `wandb` Run ID (3iv8x5fq):"
    )
    mo.vstack([sampling_radius_mm_input, rotation_input, run_id_input])
    return rotation_input, run_id_input, sampling_radius_mm_input


@app.function
def get_model(run_id):
    """
    Loads the RadiographyEncoder model from a specified checkpoint path.
    """
    if not run_id:
        return None, "Please enter a valid wandb Run ID."

    # Construct the path to the checkpoint file
    checkpoint_path = f"/cbica/home/gangarav/checkpoints/{run_id}/last.ckpt"

    if not os.path.exists(checkpoint_path):
        return None, mo.md(f"**Error**: Checkpoint not found at `{checkpoint_path}`.")

    try:
        # Load the model using the class method from the training script.
        # Lightning automatically handles hyperparameters.
        model = RadiographyEncoder.load_from_checkpoint(checkpoint_path=checkpoint_path)
        model.eval()  # Set the model to evaluation mode

        # Move model to GPU if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        status_message = mo.md(f"✅ Model from run `{run_id}` loaded successfully on `{device}`.")
        return model, status_message
    except Exception as e:
        return None, mo.md(f"**Error**: Failed to load model. Reason: {e}")


@app.cell
def _(run_id_input):
    model, message = get_model(run_id_input.value)
    message
    return (model,)


@app.cell
def _(base_patch_shape, model, n_patches_input, sampling_radius_mm, topcow_df):
    def _():
        all_embeddings = []
        all_locations = []
        all_scans = []
        centers = []

        validation_dataset = ValidationDataset(
            metadata=topcow_df,
            patch_shape=int(base_patch_shape.value),
            sampling_radius_mm=sampling_radius_mm,
            n_patches=int(n_patches_input.value),
            n_samples=1,
            n_randoms=1
        )

        loader = DataLoader(
            validation_dataset,
            batch_size=64,
            num_workers=4,
            pin_memory=True
        )

        for patches, patch_coords_vox, patch_coords_pt, label, name, subset_center in loader:
            with torch.no_grad():
                embeddings = model.encoder(patches.to(model.device), patch_coords_pt.to(model.device))
                all_embeddings.append(embeddings[:, 1])
                all_locations.extend(label)
                all_scans.extend(name)
                centers.extend(subset_center)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings, all_locations, all_scans, centers

    if model:
        all_embeddings, all_locations, all_scans, center = _()
    return all_embeddings, all_locations, all_scans, center


@app.cell
def _(all_embeddings):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    # Convert embeddings to numpy and scale them
    embeddings_np = all_embeddings.detach().cpu().numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings_np)
    return SVC, X_scaled, train_test_split


@app.cell
def _(SVC, X_scaled, all_locations, all_scans, center, train_test_split):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import imageio
    import io # <-- Import the io module

    images = []
    accuracies = []
    test_sizes = np.arange(0.1, 0.9, 0.1)

    # Iterate through different test sizes
    for test_size in test_sizes:
        # Create a fresh figure and axes for each plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Split data
        X_train, X_test, y_train, y_test, zarr_train, zarr_test, centers_train, centers_test = train_test_split(
            X_scaled, all_locations, all_scans, center, test_size=test_size, random_state=42, stratify=all_locations
        )

        # Train SVM classifier
        svm = SVC(kernel='linear', C=0.001)
        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        accuracies.append(accuracy)

        # A convenience method to plot directly from predictions

        ConfusionMatrixDisplay.from_predictions(
            y_test, 
            y_pred, 
            ax=ax, 
            labels=np.unique(all_locations), 
            normalize='true',
            im_kw={'vmin': 0, 'vmax': 1} # Pass as a dictionary here
        )
        ax.set_title(f'Test Size: {test_size:.1f}')

        # Rotate x-axis labels and give more space to y-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), fontsize=8)
        fig.subplots_adjust(left=0.2, bottom=0.2)

        # --- THE NEW, MORE ROBUST FIX IS HERE ---
        # 1. Create an in-memory buffer
        buf = io.BytesIO()
        # 2. Save the figure to the buffer as a PNG
        fig.savefig(buf, format='png', bbox_inches='tight')
        # 3. Rewind the buffer's cursor to the beginning
        buf.seek(0)
        # 4. Read the PNG data from the buffer into a numpy array
        image = imageio.imread(buf)
        images.append(image)
        # --- End of new section ---

        # Close the figure to free up memory
        plt.close(fig)

    # Create and save the GIF
    imageio.mimsave('confusion_matrix_evolution.gif', images, fps=2, plugin='pillow')

    np.mean(accuracies)

    # Display the combined plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(test_sizes, accuracies, marker='o')
    ax.set_xlabel('Test Size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Test Size')
    plt.gca()
    return


@app.cell
def _(
    base_patch_shape,
    n_patches_input,
    rotation_input,
    sampling_radius_mm_input,
):
    # Create a small test metadata dataframe
    topcow_df = pd.read_parquet('../rsna25/topcow_metadata.parquet')

    if sampling_radius_mm_input.value.strip():
        sampling_radius_mm = float(sampling_radius_mm_input.value)
    else:
        sampling_radius_mm = random.uniform(20.0, 30.0)

    if rotation_input.value.strip():
        rotation_xyz_degrees = tuple(float(x) for x in rotation_input.value.split(','))
    else:
        rotation_xyz_degrees = (random.randint(-30, 30), random.randint(-30, 30), random.randint(-30, 30))

    # Initialize the dataset with small parameters for testing
    validation_dataset = ValidationDataset(
        metadata=topcow_df,
        patch_shape=int(base_patch_shape.value),
        sampling_radius_mm=sampling_radius_mm,
        n_patches=int(n_patches_input.value),
        n_samples=1,
        n_randoms=1
    )

    next_button = mo.ui.run_button(label="next example")
    mo.vstack([next_button])
    return (
        next_button,
        rotation_xyz_degrees,
        sampling_radius_mm,
        topcow_df,
        validation_dataset,
    )


@app.cell
def _(next_button, validation_dataset):
    if next_button.value:
        # Get the iterator and fetch the next sample
        iterator = iter(validation_dataset)
        patches, patch_coords_vox, patch_coords_pt, label, name, subset_center = next(iterator)
    return label, name, patch_coords_vox, patches, subset_center


@app.function
def normalize_pixels_to_range(pixel_array, w_min, w_max, out_range=(-1.0, 1.0)):
    # Ensure w_max is greater than w_min to avoid division by zero
    if w_max <= w_min:
        w_max = w_min + 1e-6

    clipped_array = np.clip(pixel_array, w_min, w_max)
    scaled_01 = (clipped_array - w_min) / (w_max - w_min)
    out_min, out_max = out_range
    return scaled_01 * (out_max - out_min) + out_min


@app.cell
def _(label):
    label
    return


@app.cell
def _(name, patch_coords_vox, patches, subset_center):
    scan_pixels = nib.as_closest_canonical(nib.load(name)).get_fdata()

    scan_px_max = scan_pixels.max()
    scan_px_min = scan_pixels.min()

    _scan_data = normalize_pixels_to_range(scan_pixels[:], scan_px_min, scan_px_max, (0, 1))

    scan_px_copy = np.stack([_scan_data, _scan_data, _scan_data], axis=-1)

    scan_px_copy[subset_center[0]-2:subset_center[0]+3,
                  subset_center[1]-2:subset_center[1]+3,
                  subset_center[2]-2:subset_center[2]+3, :] = [1, 0, 0]

    for _i, _center in enumerate(patch_coords_vox):
        # color = rgb[_i] if i < len(rgb) else [0, 1, 0]
        scan_px_copy[_center[0]-1:_center[0]+2,
                       _center[1]-1:_center[1]+2,
                       _center[2]-1:_center[2]+2, :] = [0, 1, 0] #color/255

    patch_slider = mo.ui.slider(start=0, stop=patches.shape[0], value=patches.shape[0])
    patch_slider
    return patch_slider, scan_px_copy


@app.cell
def _(patch_slider, patches):
    _patch_display = patches[patch_slider.value, :, :].cpu().numpy().copy()
    # _patch_display[0, 0] = scan_px_max
    # _patch_display[-1, -1] = scan_px_min
    mo.image(src=_patch_display, width=128)
    return


@app.cell
def _(patch_coords_vox, patch_slider, patches, scan_px_copy, subset_center):
    slice_ax = 2

    # Determine the initial value for the slider based on the dynamic slice_axis
    if patch_slider.value == patches.shape[0]:
        # Use the coordinate of the main prism center along the sliced axis
        _slider_value = subset_center[slice_ax]
    else:
        # Use the coordinate of the selected patch center along the sliced axis
        _slider_value = patch_coords_vox[patch_slider.value][slice_ax]


    # Configure the slider to move along the correct axis of the full scan volume
    scan_slider = mo.ui.slider(
        start=0,
        stop=scan_px_copy.shape[slice_ax] - 1, # Dynamically set the range
        value=_slider_value                         # Dynamically set the initial position
    )

    scan_slider
    return scan_slider, slice_ax


@app.cell
def _(scan_px_copy, scan_slider, slice_ax):
    _slicer = [slice(None)] * 3 
    _slicer[slice_ax] = scan_slider.value
    _slice_2d = scan_px_copy[tuple(_slicer)]

    mo.image(src=_slice_2d)
    return


@app.cell
def _():
    return


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
        forward_rotation_matrix=params["forward_rotation_matrix"]
    )
    return


@app.cell
def _():
    # rgb = 128 + 10*((patch_centers_patient - main_center_patient) - relative_rotated_centers)

    # scan_max = np.max(scan.nii_data)
    # scan_min = np.min(scan.nii_data)
    # scan_data = scan.normalize_pixels_to_range(scan.nii_data[:], scan_min, scan_max, (0, 1))

    # scan_data_copy = np.stack([scan_data, scan_data, scan_data], axis=-1)

    # scan_data_copy[center_point_vox[0]-2:center_point_vox[0]+3,
    #               center_point_vox[1]-2:center_point_vox[1]+3,
    #               center_point_vox[2]-2:center_point_vox[2]+3, :] = [1, 0, 0]

    # for i, center in enumerate(patch_centers_vox):
    #     color = rgb[i] if i < len(rgb) else [0, 1, 0]
    #     scan_data_copy[center[0]-1:center[0]+2,
    #                    center[1]-1:center[1]+2,
    #                    center[2]-1:center[2]+2, :] = color/255
    return


@app.cell
def _():
    # class nifti_scan():

    #     def __init__(self, path_to_scan, median, stdev, base_patch_size):
    #         self.nii_image = nib.as_closest_canonical(nib.load(path_to_scan))
    #         self.nii_data = self.nii_image.get_fdata()
    #         self.med = median
    #         self.std = stdev
    #         self.patch_shape = self.get_outlier_axis_patch_shape(base_patch_size)

    #     def train_sample(
    #         self,
    #         n_patches: int,
    #         *, # Force subsequent arguments to be keyword-only for clarity
    #         subset_center: Optional[Tuple[int, int, int]] = None,
    #         sampling_radius_mm: Optional[int] = None,
    #         rotation_degrees: Optional[Tuple[int, int, int]] = None,
    #         wc: Optional[float] = None,
    #         ww: Optional[float] = None
    #     ) -> Dict[str, Any]:
    #         results = {}

    #         if wc is None or ww is None:
    #             wc, ww = self.get_random_wc_ww_for_scan()
    #         results['wc'], results['ww'] = wc, ww
    #         results['w_min'], results['w_max'] = wc - 0.5 * ww, wc + 0.5 * ww

    #         if sampling_radius_mm is None:
    #             sampling_radius_mm = random.uniform(20.0, 30.0)
    #         results['sampling_radius_mm'] = sampling_radius_mm

    #         if subset_center is None:
    #             subset_center = self.get_random_center_idx(sampling_radius_mm, self.patch_shape)
    #         results['subset_center'] = subset_center

    #         if rotation_degrees is None:
    #             rotation_degrees = (random.randint(-30, 30), random.randint(-30, 30), random.randint(-30, 30))
    #         results['rotation_degrees'] = rotation_degrees

    #         patches, prism_center_pt, patch_centers_vox, patch_centers_pt, relative_rotated_patch_centers_pt = self.sample_and_rotate_patches(
    #             n_patches,
    #             subset_center,
    #             sampling_radius_mm,
    #             self.patch_shape,
    #             rotation_degrees
    #         )

    #         results['prism_center_pt'] = prism_center_pt
    #         results['patch_centers_vox'] = patch_centers_vox
    #         results['patch_centers_pt'] = patch_centers_pt
    #         results['relative_patch_centers_pt'] = relative_rotated_patch_centers_pt
    #         results['raw_patches'] = patches

    #         normalized_patches = self.normalize_pixels_to_range(
    #             patches, results['w_min'], results['w_max']
    #         )

    #         results['normalized_patches'] = normalized_patches

    #         return results

    #     def get_outlier_axis_patch_shape(
    #         self, base_patch_size: int, similarity_threshold: float = 1.5
    #     ) -> Tuple[int, int, int]:
    #         """
    #         Determines patch shape by setting the scan's most distinct ("outlier") axis to 1.

    #         The outlier axis is the dimension that is furthest from the other two. It can
    #         be either the longest or the shortest. If all dimensions are similar, a random
    #         axis is chosen for slicing.

    #         Args:
    #             base_patch_size: The size for the two non-outlier axes (e.g., 16).
    #             similarity_threshold: The max/min ratio to consider dimensions similar.

    #         Returns:
    #             A (D, H, W) patch shape tuple.
    #         """
    #         shape = np.array(self.nii_data.shape)

    #         # 1. If dimensions are similar, pick a random axis for slicing
    #         if max(shape) / min(shape) < similarity_threshold:
    #             outlier_axis_idx = random.randint(0, 2)
    #         else:
    #             # 2. Find the outlier axis by finding the dimension that is furthest
    #             #    from the mean of the other two.
    #             diffs = [
    #                 abs(shape[0] - (shape[1] + shape[2]) / 2.0), # How far axis 0 is from the mean of 1 and 2
    #                 abs(shape[1] - (shape[0] + shape[2]) / 2.0), # How far axis 1 is from the mean of 0 and 2
    #                 abs(shape[2] - (shape[0] + shape[1]) / 2.0)  # How far axis 2 is from the mean of 0 and 1
    #             ]
    #             outlier_axis_idx = np.argmax(diffs)

    #         # 3. Construct the patch shape
    #         patch_shape = [base_patch_size] * 3
    #         patch_shape[outlier_axis_idx] = 1

    #         return tuple(patch_shape)


    #     def get_random_wc_ww_for_scan(self):
    #         return random.uniform(self.med-self.std, self.med+self.std), random.uniform(2*self.std, 6*self.std)

    #     def get_random_center_idx(self, sampling_radius_mm, patch_shape_mm):
    #         """
    #         Finds a random center index using a simplified heuristic (1.5x patch shape)
    #         to define the boundary buffer.
    #         """
    #         voxel_spacing = np.array(self.nii_image.header.get_zooms()[:3])

    #         # --- 1. Calculate buffer for the patch using the 1.5x heuristic ---
    #         # Convert patch shape from mm to the original, anisotropic voxel units.
    #         patch_shape_vox = np.ceil(np.array(patch_shape_mm) / voxel_spacing)
    #         # The buffer is half of the patch size, scaled by 1.5 to approximate the diagonal.
    #         patch_buffer_vox = np.ceil(patch_shape_vox * 1.5 / 2.0).astype(int)

    #         # --- 2. Calculate buffer needed for the sampling radius ---
    #         sampling_radius_vox = np.ceil(sampling_radius_mm / voxel_spacing).astype(int)

    #         # --- 3. Combine buffers to define the safe sampling zone ---
    #         total_buffer_vox = patch_buffer_vox + sampling_radius_vox

    #         min_idx = total_buffer_vox
    #         max_idx = np.array(self.nii_data.shape) - total_buffer_vox - 1

    #         # Ensure we have a valid range to sample from.
    #         if np.any(max_idx < min_idx):
    #             raise ValueError(
    #                 "Sampling radius and/or patch shape are too large for the given image dimensions."
    #             )

    #         # Generate a random index within the safe range.
    #         random_idx = np.array(
    #             [np.random.randint(low, high + 1) for low, high in zip(min_idx, max_idx)]
    #         )
    #         return random_idx

    #     def normalize_pixels_to_range(self, pixel_array, w_min, w_max, out_range=(-1.0, 1.0)):
    #         # Ensure w_max is greater than w_min to avoid division by zero
    #         if w_max <= w_min:
    #             w_max = w_min + 1e-6

    #         clipped_array = np.clip(pixel_array, w_min, w_max)
    #         scaled_01 = (clipped_array - w_min) / (w_max - w_min)
    #         out_min, out_max = out_range
    #         return scaled_01 * (out_max - out_min) + out_min

    #     @staticmethod
    #     def sample_patch_centers(
    #         center_point_vox: np.ndarray,
    #         sampling_radius_mm: float,
    #         num_patches: int,
    #         voxel_spacing: np.ndarray,
    #         volume_shape: Tuple[int, int, int]
    #     ) -> np.ndarray:
    #         """
    #         Samples patch center coordinates within a sphere using rejection sampling.

    #         Returns:
    #             np.ndarray: An array of shape (num_patches, 3) with patch centers in voxel coordinates.
    #         """
    #         center_point_mm = np.array(center_point_vox) * voxel_spacing
    #         valid_centers_vox = []
    #         while len(valid_centers_vox) < num_patches:
    #             needed = num_patches - len(valid_centers_vox)
    #             points_to_sample = int(needed * 2.5) + 20 # Oversample to be safe

    #             # 1. Uniformly sample within a bounding box
    #             low_bound_mm = center_point_mm - sampling_radius_mm
    #             high_bound_mm = center_point_mm + sampling_radius_mm
    #             sampled_points_mm = np.random.uniform(
    #                 low=low_bound_mm, high=high_bound_mm, size=(points_to_sample, 3)
    #             )

    #             # 2. Reject samples outside the sphere
    #             distances_sq = np.sum((sampled_points_mm - center_point_mm)**2, axis=1)
    #             in_sphere_mask = distances_sq <= sampling_radius_mm**2
    #             sampled_points_mm = sampled_points_mm[in_sphere_mask]

    #             # 3. Convert to voxel coordinates and reject if outside volume
    #             sampled_points_vox = np.round(sampled_points_mm / voxel_spacing).astype(int)
    #             in_volume_mask = np.all(sampled_points_vox >= 0, axis=1) & np.all(sampled_points_vox < volume_shape, axis=1)
    #             valid_centers_vox.extend(sampled_points_vox[in_volume_mask])

    #         return np.array(valid_centers_vox[:num_patches])

    #     @staticmethod
    #     def convert_voxel_to_patient(points_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    #         """Converts voxel coordinates to patient/world coordinates (mm)."""
    #         points_vox = np.atleast_2d(points_vox)
    #         homogeneous_coords = np.hstack((points_vox, np.ones((points_vox.shape[0], 1))))
    #         patient_coords = (affine @ homogeneous_coords.T).T[:, :3]
    #         return patient_coords.squeeze()

    #     @staticmethod
    #     def calculate_extraction_parameters(
    #         patch_shape_mm: Tuple[int, int, int],
    #         rotation_xyz_degrees: Tuple[float, float, float],
    #         voxel_spacing: np.ndarray
    #     ) -> Dict[str, Any]:
    #         """Calculates all shapes, matrices, and factors needed for patch extraction."""
    #         final_patch_shape_iso_vox = np.ceil(patch_shape_mm).astype(int)

    #         rotation = Rotation.from_euler('xyz', rotation_xyz_degrees, degrees=True)
    #         forward_rotation_matrix = rotation.as_matrix()
    #         inverse_rotation_matrix = np.linalg.inv(forward_rotation_matrix)

    #         rotated_bbox_dims = np.abs(forward_rotation_matrix) @ final_patch_shape_iso_vox
    #         source_block_shape_iso_vox = np.ceil(rotated_bbox_dims).astype(int) + 3

    #         resampling_factor = voxel_spacing / np.array([1.0, 1.0, 1.0])
    #         source_block_shape_orig_vox = np.ceil(source_block_shape_iso_vox / resampling_factor).astype(int)

    #         return {
    #             "final_patch_shape_iso_vox": final_patch_shape_iso_vox,
    #             "source_block_shape_iso_vox": source_block_shape_iso_vox,
    #             "source_block_shape_orig_vox": source_block_shape_orig_vox,
    #             "forward_rotation_matrix": forward_rotation_matrix,
    #             "inverse_rotation_matrix": inverse_rotation_matrix,
    #         }

    #     @staticmethod
    #     def extract_single_patch(
    #         data_volume: np.ndarray,
    #         center_orig_vox: np.ndarray,
    #         params: Dict[str, Any]
    #     ) -> np.ndarray:
    #         """Extracts, resamples, rotates, and crops a single patch from the volume."""
    #         # --- 1. Safe Extraction from original volume ---
    #         starts = center_orig_vox - params['source_block_shape_orig_vox'] // 2
    #         ends = starts + params['source_block_shape_orig_vox']
    #         source_block = np.zeros(params['source_block_shape_orig_vox'], dtype=data_volume.dtype)

    #         crop_starts = np.maximum(starts, 0)
    #         crop_ends = np.minimum(ends, data_volume.shape)
    #         paste_starts = crop_starts - starts
    #         paste_ends = paste_starts + (crop_ends - crop_starts)

    #         source_block[tuple(slice(s, e) for s, e in zip(paste_starts, paste_ends))] = \
    #             data_volume[tuple(slice(s, e) for s, e in zip(crop_starts, crop_ends))]

    #         # --- 2. Resample to Isotropic ---
    #         zoom_factor = params['source_block_shape_iso_vox'] / source_block.shape
    #         isotropic_block = ndimage.zoom(source_block, zoom_factor, order=1, mode='constant', cval=0.0)

    #         # --- 3. Rotate ---
    #         block_center = (np.array(isotropic_block.shape) - 1) / 2.0
    #         offset = block_center - params['inverse_rotation_matrix'] @ block_center
    #         rotated_block = ndimage.affine_transform(isotropic_block, params['inverse_rotation_matrix'], offset=offset, order=1)

    #         # --- 4. Crop final patch from center ---
    #         rot_center = (np.array(rotated_block.shape) - 1) / 2.0
    #         crop_starts = np.round(rot_center - (np.array(params['final_patch_shape_iso_vox']) / 2.0)).astype(int)
    #         slicer = tuple(slice(s, s + d) for s, d in zip(crop_starts, params['final_patch_shape_iso_vox']))

    #         return rotated_block[slicer]

    #     @staticmethod
    #     def calculate_rotated_relative_positions(
    #         patch_centers_patient: np.ndarray,
    #         main_center_patient: np.ndarray,
    #         forward_rotation_matrix: np.ndarray
    #     ) -> np.ndarray:
    #         """Calculates the final rotated relative position vectors."""
    #         relative_vectors = patch_centers_patient - main_center_patient
    #         return (forward_rotation_matrix @ relative_vectors.T).T

    #     def sample_and_rotate_patches(
    #         self,
    #         num_patches,
    #         center_point,
    #         sampling_radius_mm,
    #         patch_shape_mm,
    #         rotation_xyz_degrees=(0, 0, 0)
    #     ):
    #         """
    #         Orchestrates the patch extraction process using modular static helper functions.
    #         """
    #         # --- Step 0: Get instance properties ---
    #         affine = np.array(self.nii_image.affine)
    #         voxel_spacing = np.array(self.nii_image.header.get_zooms()[:3])

    #         # --- Step 1: Sample patch center locations in original voxel space ---
    #         patch_centers_vox = self.sample_patch_centers(
    #             center_point_vox=center_point,
    #             sampling_radius_mm=sampling_radius_mm,
    #             num_patches=num_patches,
    #             voxel_spacing=voxel_spacing,
    #             volume_shape=self.nii_data.shape
    #         )

    #         # --- Step 2: Convert key center points from voxel to patient space (mm) --
    #         main_center_patient = self.convert_voxel_to_patient(center_point, affine)
    #         patch_centers_patient = self.convert_voxel_to_patient(patch_centers_vox, affine)

    #         # --- Step 3: Pre-calculate all shapes, matrices, and other parameters ---
    #         params = self.calculate_extraction_parameters(
    #             patch_shape_mm=patch_shape_mm,
    #             rotation_xyz_degrees=rotation_xyz_degrees,
    #             voxel_spacing=voxel_spacing
    #         )

    #         # --- Step 4: Loop through centers to extract and process each patch ---
    #         final_patches = np.array([
    #             self.extract_single_patch(
    #                 data_volume=self.nii_data,
    #                 center_orig_vox=center_vox,
    #                 params=params
    #             ) for center_vox in patch_centers_vox
    #         ])

    #         final_patches = np.squeeze(final_patches)

    #         # --- Step 5: Calculate the final rotated relative position vectors ---
    #         relative_rotated_centers = self.calculate_rotated_relative_positions(
    #             patch_centers_patient=patch_centers_patient,
    #             main_center_patient=main_center_patient,
    #             forward_rotation_matrix=params["forward_rotation_matrix"]
    #         )

    #         # --- Step 6: Return all results ---
    #         return final_patches, main_center_patient, patch_centers_vox, patch_centers_patient, relative_rotated_centers
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
