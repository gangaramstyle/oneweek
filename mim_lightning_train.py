import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import wandb
    import sys
    import os
    import glob
    import shutil
    import torch
    from torch import optim, nn, utils, Tensor
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    import lightning as L
    from pytorch_lightning.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from rvt_model import RvT, PosEmbedding3D
    from torch.utils.data import DataLoader, IterableDataset
    from typing import Optional, Tuple, Dict, Any
    from data_loader import nifti_scan
    from x_transformers import CrossAttender
    import pandas as pd
    import numpy as np
    import random
    import time
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    import tempfile
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    import matplotlib
    matplotlib.use('Agg') # <-- This is correct, keep it.

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import imageio
    import io

    from types import SimpleNamespace


@app.class_definition
# define the LightningModule
class RadiographyEncoder(L.LightningModule):

    class MIMHead(nn.Module):
        def __init__(self, dim, pos_max_freq, patch_size, depth, layer_dropout):
            super().__init__()
            self.pos_emb = PosEmbedding3D(dim, max_freq = pos_max_freq)
            self.mim = CrossAttender(dim = dim, depth = depth, layer_dropout=layer_dropout)
            self.to_pixels = nn.Linear(dim, np.prod(patch_size))
            self.patch_size = patch_size

        def forward(self, coords, context):
            sin, cos = self.pos_emb(coords)
            sin_unrepeated = sin[:, :, 0::2]
            cos_unrepeated = cos[:, :, 0::2]
            pos = PosEmbedding3D.interleave_two_tensors(sin_unrepeated, cos_unrepeated)

            unmasked = self.mim(pos, context=context)
            unmasked = self.to_pixels(unmasked)
            unmasked = unmasked.view(unmasked.size(0), unmasked.size(1), *self.patch_size)
            return unmasked

    @staticmethod
    def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
        """
        Computes the Supervised Contrastive Loss for a batch of embeddings.

        Args:
            embeddings (torch.Tensor): A tensor of shape [N, D] where N is the batch size
                                       and D is the embedding dimension. Embeddings should be
                                       L2 normalized.
            labels (torch.Tensor): A tensor of shape [N] with the sample ID for each embedding.
            temperature (float): The temperature scaling factor.

        Returns:
            torch.Tensor: The calculated loss.
        """
        device = embeddings.device
        n = embeddings.shape[0]

        # 1. Calculate all-pairs similarity
        # The result is a matrix of shape [N, N]
        similarity_matrix = embeddings @ embeddings.t()

        # 2. Create the positive-pair mask
        # The mask will be True where labels are the same, False otherwise.
        # labels.unsqueeze(0) creates a row vector [1, N]
        # labels.unsqueeze(1) creates a column vector [N, 1]
        # Broadcasting them results in a [N, N] matrix of pairwise label comparisons.
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

        # 3. Discard self-similarity from positives
        # We create a mask to remove the diagonal (where an embedding is compared to itself)
        # an embedding cannot be its own positive pair.
        identity_mask = torch.eye(n, device=device).bool()
        positives_mask = labels_matrix & ~identity_mask

        # 4. Create mask for negative pairs
        # Negatives are all pairs that are not self and not positive.
        negatives_mask = ~labels_matrix

        # The original NT-Xent loss can be simplified for this multi-positive case.
        # For each anchor, the loss is -log( sum(exp(sim_pos)) / sum(exp(sim_all_others)) )

        # To prevent log(0) issues for anchors with no positive pairs, we can mask them out.
        # However, the formulation below handles this gracefully.

        # We need a mask to exclude the diagonal from the denominator's log-sum-exp
        logits_mask = ~identity_mask

        # Apply temperature scaling
        similarity_matrix /= temperature

        # For each row (anchor), we compute the log-softmax over all other samples.
        # The similarity_matrix[logits_mask] flattens the matrix, removing the diagonal.
        # .reshape(n, n - 1) makes it a [N, N-1] matrix where each row corresponds
        # to the similarities of one anchor to all N-1 other samples.
        log_probs = nn.functional.log_softmax(similarity_matrix[logits_mask].reshape(n, n - 1), dim=1)

        # The positives_mask now needs to align with the log_probs matrix.
        # We remove the diagonal from positives_mask as well.
        positives_mask_for_loss = positives_mask[logits_mask].reshape(n, n - 1)

        # For each anchor, we want to sum the log-probabilities of its positive pairs.
        # We use the positive mask to select these probabilities.
        # We normalize by the number of positive pairs for each anchor to get the mean.
        # Adding a small epsilon (1e-7) to the denominator prevents division by zero
        # in case an anchor has no positive pairs (it's the only one of its class).
        num_positives_per_row = positives_mask_for_loss.sum(dim=1)
        loss = - (positives_mask_for_loss * log_probs).sum(dim=1) / (num_positives_per_row + 1e-7)

        # We average the loss over all anchors that had at least one positive pair.
        # This prevents anchors with no positives from contributing a 0 to the mean.
        loss = loss[num_positives_per_row > 0].mean()

        return loss

    def __init__(
        self,
        *,
        # model hyperparameters
        encoder_dim,
        encoder_depth,
        encoder_heads,
        mlp_dim,
        n_registers,
        # position embedding
        pos_max_freq,
        use_rotary,
        use_absolute,
        # training runtime hyperparameters
        batch_size,
        learning_rate,
        # dataset hyperparameters
        patch_size,
        patch_jitter,
        # objectives
        pos_objective_mode,
        window_objective,
        scan_contrastive_objective,
        mim_objective,

    ):
        # view objectives refer to:
        #  x, y, z axes
        #  (window width, window center)
        #  (zoom scale, rotation x, y, z)
        self.NUM_POS_OBJECTIVES = 6
        self.NUM_WINDOW_OBJECTIVES = 2
        self.n_registers = n_registers

        super().__init__()

        self.save_hyperparameters()
        self.encoder = RvT(
            patch_size=patch_size,
            register_count=n_registers,
            dim=encoder_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            mlp_dim=mlp_dim,
            use_rotary=use_rotary,
            pos_max_freq=pos_max_freq,
            use_absolute=use_absolute,
        )

        # CAUTION: relative view head requires two concatenated
        # encoder outputs because it calculates the relative
        # difference* between the objectives
        self.relative_pos_head = nn.Sequential(
            nn.LayerNorm(encoder_dim * 2), nn.Linear(encoder_dim * 2, self.NUM_POS_OBJECTIVES)
        )

        self.relative_window_head = nn.Sequential(
            nn.LayerNorm(encoder_dim * 2), nn.Linear(encoder_dim * 2, self.NUM_WINDOW_OBJECTIVES)
        )

        if self.hparams.pos_objective_mode == "classification":
            self.pos_view_criterion = nn.BCEWithLogitsLoss()
        elif self.hparams.pos_objective_mode == "regression":
            self.pos_view_criterion = nn.MSELoss()

        self.window_view_criterion = nn.BCEWithLogitsLoss()

        self.mim = RadiographyEncoder.MIMHead(encoder_dim, pos_max_freq, patch_size, depth=4, layer_dropout=0.2)
        self.mim_loss = nn.SmoothL1Loss()

        self.validation_step_outputs = []

    def raw_encoder_emb_to_scan_view_registers_patches(self, emb):
        # 0: global scan embedding
        # 1: local view embedding
        # 2 - 2+registers: register tokens
        # 2+registers - end: patch embeddings
        return (
            emb[:, 0:1],
            emb[:, 1:2],
            emb[:, 2 : 2 + self.n_registers],
            emb[:, 2 + self.n_registers :],
        )

    def training_step(self, batch, batch_idx):
        patches_1, patches_2, patch_coords_1, patch_coords_2, label = batch

        # Split patches and indices for input vs label
        split_size = 50

        input_patches_1 = patches_1[:,:split_size]
        mim_patches_1 = patches_1[:,split_size:]
        input_coords_1 = patch_coords_1[:,:split_size]
        mim_coords_1 = patch_coords_1[:,split_size:]

        input_patches_2 = patches_2[:,:split_size]
        mim_patches_2 = patches_2[:,split_size:]
        input_coords_2 = patch_coords_2[:,:split_size]
        mim_coords_2 = patch_coords_2[:,split_size:]

        emb1 = self.encoder(input_patches_1, input_coords_1)
        emb2 = self.encoder(input_patches_2, input_coords_2)

        scan_cls_1, view_cls_1, registers_1, patch_emb_1 = (
            self.raw_encoder_emb_to_scan_view_registers_patches(emb1)
        )
        scan_cls_2, view_cls_2, registers_2, patch_emb_2 = (
            self.raw_encoder_emb_to_scan_view_registers_patches(emb2)
        )

        fused_view_cls = torch.cat((view_cls_1.squeeze(), view_cls_2.squeeze()), dim=1)

        # fused_scan_cls = torch.cat((scan_cls_1.squeeze(), scan_cls_2.squeeze()), dim = 0)
        # fused_scan_cls = nn.functional.normalize(fused_scan_cls, p=2, dim=1)
        # fused_scan_ids = torch.cat((row_id, row_id), dim = 0)

        # --- DYNAMIC LOSS CALCULATION ---
        total_loss = 0.0

        pos_label = label[:, :self.NUM_POS_OBJECTIVES]
        if self.hparams.pos_objective_mode == "classification":
            pos_target = (pos_label > 0).to(torch.float32)
        elif self.hparams.pos_objective_mode == "regression":
            # Target is the actual distance value
            pos_target = (pos_label / 100).to(torch.float32)

        pos_prediction = self.relative_pos_head(fused_view_cls)

        # --- Calculate total position loss for backpropagation ---
        pos_loss = self.pos_view_criterion(pos_prediction, pos_target)

        # --- Add component losses for logging ---
        # Translation loss (first 3 elements)
        translation_loss = self.pos_view_criterion(pos_prediction[:, :3], pos_target[:, :3])
        self.log("translation_loss", translation_loss)

        # Rotation loss (next 3 elements: 3, 4, 5)
        rotation_loss = self.pos_view_criterion(pos_prediction[:, 3:6], pos_target[:, 3:6])
        self.log("rotation_loss", rotation_loss)

        # --- Update total loss and log the combined position loss ---
        total_loss += pos_loss
        self.log("pos_loss", pos_loss)

        if self.hparams.window_objective:
            window_label = label[:, -self.NUM_WINDOW_OBJECTIVES:]
            window_target = (window_label > 0).to(torch.float32)

            window_prediction = self.relative_window_head(fused_view_cls)
            window_loss = self.window_view_criterion(window_prediction, window_target)

            total_loss += window_loss
            self.log("window_loss", window_loss)

        # if self.hparams.scan_contrastive_objective:
        #     scan_loss = self.supervised_contrastive_loss(fused_scan_cls, fused_scan_ids)/4.0
        #     total_loss += scan_loss
        #     self.log("scan_loss", scan_loss)

        if self.hparams.mim_objective:
            mim_prediction_1 = self.mim(
                mim_coords_1,
                torch.cat((scan_cls_1, view_cls_1, patch_emb_1), dim=1)
            )
            mim_prediction_2 = self.mim(
                mim_coords_2,
                torch.cat((scan_cls_2, view_cls_2, patch_emb_2), dim=1)
            )

            mim_1_loss = self.mim_loss(mim_prediction_1, mim_patches_1)
            mim_2_loss = self.mim_loss(mim_prediction_2, mim_patches_2)

            self.log("mim_loss", mim_1_loss + mim_2_loss)
            total_loss += mim_1_loss + mim_2_loss

        self.log("loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # The validation dataloader yields (patches, centers, location)
        patches, patch_coords, locations, _, _ = batch

        # Get the view embedding
        emb = self.encoder(patches, patch_coords)
        view_embedding = emb[:, 1]

        # Store the outputs for later use in `on_validation_epoch_end`
        # .detach().cpu() is important to avoid GPU memory leaks
        output = {"embeddings": view_embedding.detach().cpu(), "locations": locations}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            print("No validation outputs to process.")
            return

        # --- 1. Aggregate all embeddings and locations from batches ---
        all_embeddings = torch.cat([x["embeddings"] for x in self.validation_step_outputs]).numpy()

        # Locations might be a list of tuples, so we flatten it
        all_locations = []
        for x in self.validation_step_outputs:
            all_locations.extend(x["locations"])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(all_embeddings)

        images = []
        accuracies = []
        test_sizes = np.arange(0.1, 1.0, 0.1)

        # Iterate through different test sizes
        for test_size in test_sizes:
            # Create a fresh figure and axes for each plot
            fig, ax = plt.subplots(figsize=(10, 8))

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, all_locations, test_size=test_size, random_state=42, stratify=all_locations
            )

            # Train SVM classifier
            svm = SVC(kernel='linear', C=0.001, class_weight='balanced')
            svm.fit(X_train, y_train)

            y_pred = svm.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            accuracies.append(accuracy)

            # A convenience method to plot directly from predictions

            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, normalize='true')
            ax.images[0].set_clim(0, 1)
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
            plt.close(fig)

        # Log the mean accuracy
        svm_accuracy = np.mean(accuracies)
        self.log("svm_accuracy", svm_accuracy, prog_bar=True)
        print(f"svm_accuracy: {svm_accuracy:.4f}")

        # Define a path for the GIF
        gif_path = "confusion_matrix_evolution.gif"

        # Save the GIF to the path
        imageio.mimsave(gif_path, images, fps=2, plugin='pillow')

        # Log the GIF file to W&B as a video
        self.logger.experiment.log({
            "svm_confusion_matrix": wandb.Video(gif_path, fps=2, format="gif")
        })

        # Clean up the outputs list for the next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Use the learning_rate from hparams so it can be configured by sweeps
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


@app.cell
def _(sample_1_aux, sample_2_aux):
    class PrismOrderingDataset(IterableDataset):

        def __init__(self, metadata, include_nifti, patch_shape, position_space, n_patches, n_aux_patches, n_sampled_from_same_study, scratch_dir):
            super().__init__()
            self.metadata = pd.read_parquet(metadata).dropna()
            self.patch_shape = patch_shape
            self.n_patches = n_patches
            self.n_sampled_from_same_study = n_sampled_from_same_study
            self.position_space=position_space


        def generate_training_pair(self, scan, n_patches: int, position_space = "patient", n_aux_patches: int = 0, debug: bool = False) -> tuple:
            """
            A high-level helper for training loops that generates a pair of samples.
            """

            wc1, ww1 = scan.get_random_wc_ww_for_scan()
            wc2, ww2 = scan.get_random_wc_ww_for_scan()

            sample_1_data = scan.train_sample(n_patches=n_patches, patch_shape=(16, 16, 1), wc=wc1, ww=ww1)
            sample_2_data = scan.train_sample(n_patches=n_patches, patch_shape=(16, 16, 1), wc=wc2, ww=ww2)


            patches_1 = sample_1_data['normalized_patches']
            patches_2 = sample_2_data['normalized_patches']

            patch_coords_1 = sample_1_data['relative_patch_centers_pt']
            patch_coords_2 = sample_2_data['relative_patch_centers_pt']

            pos_label = (sample_2_data['prism_center_pt'] - sample_1_data['prism_center_pt'])

            rotation_label = (np.array(sample_2_data['rotation_degrees']) - np.array(sample_1_data['rotation_degrees']))

            # window based relative view information
            window_label = np.array([wc2 - wc1, ww2 - ww1])

            label = np.concatenate((pos_label, rotation_label, window_label))


            tensors = [
                patches_1, patches_2, patch_coords_1, patch_coords_2,
                label
            ]
            if debug:
                return tuple(torch.from_numpy(arr).to(torch.float32) for arr in tensors), sample_1_data, sample_2_data, sample_1_aux, sample_2_aux

            return tuple(torch.from_numpy(arr).to(torch.float32) for arr in tensors)

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else 0 # Get worker ID

            if worker_info is None:
                # Case: num_workers = 0. The main process gets all the data.
                worker_id = 0
                worker_metadata = self.metadata
            else:
                # Case: num_workers > 0.
                worker_id = worker_info.id
                worker_metadata = self.metadata.iloc[worker_id::worker_info.num_workers]

                # Seed each worker differently to ensure random shuffling is unique
                seed = (torch.initial_seed() + worker_id) % (2**32)
                np.random.seed(seed)
                random.seed(seed)

            print(f"[Worker {worker_id}] assigned {len(worker_metadata)} scans.")

            while True:

                sample = worker_metadata.sample(n=1).iloc[0]

                path_to_load = (
                    sample["zarr_path"]
                    .replace(
                        "/cbica/home/gangarav/data_25_processed/",
                        "/cbica/home/gangarav/rsna_any/nifti/",
                    )
                    .replace(".zarr", ".nii")
                )

                median = sample["median"]
                stdev = sample["stdev"]

                # 2. Instantiate the scan loader with all necessary info
                try:
                    print(path_to_load)

                    scan = nifti_scan(
                        path_to_scan=path_to_load,
                        median=median,
                        stdev=stdev,
                    )

                    print(f"[Worker {worker_id}] Generating pairs for {path_to_load}...")
                    for _ in range(self.n_sampled_from_same_study):
                        training_pair = self.generate_training_pair(
                            scan,
                            n_patches=self.n_patches,
                        )

                        yield training_pair

                except (ValueError, FileNotFoundError) as e:
                    print(f"[Worker {worker_id}] CRITICAL: Skipping scan {path_to_load} due to error: {e}")
                    continue
    return (PrismOrderingDataset,)


@app.cell
def _(zarr_scan):
    class ValidationDataset(IterableDataset):
        def __init__(self, metadata, prism_shape=(6, 64, 64), patch_shape=None, n_patches=None):
            super().__init__()
            metadata_df = pd.read_parquet(metadata)
            aneurysm_df = pd.read_parquet("/cbica/home/gangarav/rsna25/aneurysm_labels_with_nifti_coords.parquet")

            aneurysm_subset = aneurysm_df[['SeriesInstanceUID', 'location', 'modality', 'image_position_delta_X', 'image_position_delta_Y', 'image_position_delta_Z', 'pixel_x', 'pixel_y', 'pixel_z']]
            metadata_df = metadata_df.merge(aneurysm_subset, left_on='series_uid', right_on='SeriesInstanceUID', how='inner')
            self.metadata = metadata_df.drop(columns=['modality_x'])
            self.prism_shape = prism_shape
            self.patch_shape = patch_shape
            self.n_patches = n_patches
            print(f"Initialized validation dataset with {len(self.metadata)} samples.")

        def __iter__(self):
            # No need for worker splitting if num_workers=0, which is typical for smaller validation sets
            for _, row in self.metadata.iterrows():
                zarr_name = row["zarr_path"]

                # patch_shape = (1, patch_size, patch_size)
                scan = zarr_scan(path_to_scan=zarr_name, median=row["median"], stdev=row["stdev"],  patch_shape=self.patch_shape)
                for i in range(1):
                    sample = scan.train_sample(self.n_patches, subset_center=(row["pixel_x"], row["pixel_y"], row["pixel_z"]))


                    patches = torch.from_numpy(sample["normalized_patches"]).to(torch.float32)
                    patch_coords = torch.from_numpy(sample['patch_centers_pt'] - sample['subset_center_pt']).to(torch.float32)

                    # Yield data in the format expected by validation_step
                    yield patches, patch_coords, row["location"], zarr_name, row["modality_y"]

                for i in range(1):
                    sample = scan.train_sample(self.n_patches)


                    patches = torch.from_numpy(sample["normalized_patches"]).to(torch.float32)
                    patch_coords = torch.from_numpy(sample['patch_centers_pt'] - sample['subset_center_pt']).to(torch.float32)

                    # Yield data in the format expected by validation_step
                    yield patches, patch_coords, "random", zarr_name, row["modality_y"]
    return


@app.cell
def _(PrismOrderingDataset):
    def get_allocated_cpus():
        """
        Gets the number of CPUs allocated to the job.
        It checks for common environment variables set by cluster schedulers.
        If not found, it falls back to the total number of CPUs on the machine.
        """
        # Check for Slurm
        return int(os.environ.get("SLURM_CPUS_PER_GPU", 8))

    def get_gpu_memory_gb():
        """
        Gets the total memory of the first available GPU in gigabytes.
        Returns a dictionary where keys are device IDs and values are memory in GB.
        Returns an empty dictionary if no GPU is found.
        """
        gpu_memory = {}
        if not torch.cuda.is_available():
            return gpu_memory

        for i in range(torch.cuda.device_count()):
            total_mem_bytes = torch.cuda.get_device_properties(i).total_memory
            total_mem_gb = round(total_mem_bytes / (1024**3), 2) # Convert bytes to GiB
            gpu_memory[i] = total_mem_gb
        return gpu_memory

    def train_run(default_config=None):
        """
        Main training function that can be called by a sweep agent or for a single run.
        Handles both starting new runs and resuming existing ones.
        """
        # --- 1. Initialize Weights & Biases ---
        run = wandb.init(resume="allow")

        print(f"Run config:\n{run.config}")
        print("="*20)

        # We will still use a local checkpoint directory for new checkpoints
        checkpoint_dir = f'../checkpoints/{run.id}'

        # Pull the final config from wandb.
        config_dict = {
            "batch_size": 256,
            "encoder_depth": 16,
            "encoder_dim": 432,
            "encoder_heads": 12,
            "include_nifti": True,
            "learning_rate": 0.00008232689121870076,
            "max_freq": 60,
            "mim_objective": True,
            "mlp_dim": 768,
            "n_aux_patches": 0,
            "n_registers": 8,
            "num_repeated_study_samples": 64,
            "patch_jitter": 1,
            "patch_size": (16, 16, 1),
            "pos_max_freq": 60,
            "pos_objective_mode": "classification",
            "position_space": "patient",
            "scan_contrastive_objective": False,
            "use_absolute": True,
            "use_rotary": True,
            "window_objective": False,
        }

        cfg = SimpleNamespace(**config_dict)

        # --- 2. Handle Resuming from Wandb Artifacts ---
        ckpt_path = None
        if run.resumed:
            print(f"Resuming run '{run.id}' from a wandb artifact...")
            try:
                # Construct the reference to the latest artifact for this run.
                # The WandbLogger by default names the artifact 'model-<run.id>'
                artifact_ref = f"{run.entity}/{run.project}/model-{run.id}:latest"
                print(f"Attempting to download artifact: {artifact_ref}")

                # Use the artifact and download its contents
                artifact = run.use_artifact(artifact_ref, type='model')
                artifact_dir = artifact.download()

                # The artifact directory contains the checkpoint file. We need to find it.
                # It's often named 'model.ckpt' or something similar. Using glob is robust.
                # The 'last.ckpt' symlink is not part of the artifact, so we look for the actual file.
                ckpt_files = glob.glob(f"{artifact_dir}/*.ckpt")

                if ckpt_files:
                    ckpt_path = ckpt_files[0]  # Get the first match
                    print(f"Successfully found and downloaded checkpoint: {ckpt_path}")
                else:
                    print(f"WARNING: Artifact was downloaded, but no '*.ckpt' file was found in '{artifact_dir}'. Starting from scratch.")

            except wandb.errors.CommError:
                # This error occurs if the artifact doesn't exist (e.g., run was started
                # but no checkpoint was saved yet).
                print(f"WARNING: Could not find a 'model-{run.id}:latest' artifact. "
                      "The run will start from scratch but log to the same wandb run.")

        # --- 3. Setup Model ---
        model = RadiographyEncoder(
            encoder_dim=cfg.encoder_dim,
            encoder_depth=cfg.encoder_depth,
            encoder_heads=cfg.encoder_heads,
            mlp_dim=cfg.mlp_dim,
            n_registers=cfg.n_registers,
            pos_max_freq=cfg.max_freq,
            use_rotary=cfg.use_rotary,
            use_absolute=cfg.use_absolute,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            patch_size=cfg.patch_size,
            patch_jitter=1.0,
            pos_objective_mode=cfg.pos_objective_mode,
            window_objective=cfg.window_objective,
            scan_contrastive_objective=cfg.scan_contrastive_objective,
            mim_objective=cfg.mim_objective,
        )

        # --- 4. Setup Data ---
        N_PATCHES = 64
        NUM_WORKERS = int(get_allocated_cpus())-2
        METADATA_PATH = os.environ.get(
            'METADATA_PATH', 
            '/default/path/for/local/testing.parquet'
        )
        # METADATA_PATH = '/cbica/home/gangarav/rsna25/aneurysm_labels_with_nifti_coords.parquet'

        base_temp_dir = tempfile.gettempdir()

        if not os.path.isdir(base_temp_dir) or not os.access(base_temp_dir, os.W_OK):
            raise IOError(
                f"The determined temporary directory '{base_temp_dir}' "
                "is not a writable directory. Check system configuration and permissions."
            )

        scratch_dir = os.path.join(base_temp_dir, "scans")
        os.makedirs(scratch_dir, exist_ok=True)

        dataset = PrismOrderingDataset(
            include_nifti=cfg.include_nifti,
            metadata=METADATA_PATH,
            patch_shape=cfg.patch_size,
            n_patches=N_PATCHES,
            position_space=cfg.position_space,
            n_aux_patches=cfg.n_aux_patches,
            scratch_dir=scratch_dir,
            n_sampled_from_same_study=cfg.num_repeated_study_samples
        )

        dataloader = DataLoader(
            dataset,
            batch_size=int(cfg.batch_size * (get_gpu_memory_gb()[0]/100.0)),
            num_workers=int(NUM_WORKERS),
            persistent_workers=(NUM_WORKERS > 0),
            pin_memory=True,
        )

        # val_dataset = ValidationDataset(
        #     metadata=METADATA_PATH,
        #     patch_shape=PATCH_SHAPE,
        #     n_patches=N_PATCHES
        # )
        # val_dataloader = DataLoader(
        #     val_dataset,
        #     batch_size=2*int(cfg.batch_size * (get_gpu_memory_gb()[0]/100.0)),
        #     num_workers=2,
        #     persistent_workers=True,
        #     pin_memory=True,
        # )

        wandb_logger = WandbLogger(log_model="all")

        # Checkpoints are saved in a directory named after the unique wandb run ID
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{step}',
            every_n_train_steps=2000,
            save_last=True,
        )

        # --- 6. Setup Trainer ---
        trainer = L.Trainer(
            max_epochs=-1, # For iterable datasets, steps are better than epochs
            max_steps=5000000, # Example: set a max number of steps
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=int(80.0/get_gpu_memory_gb()[0]),
            logger=wandb_logger,
            log_every_n_steps=25,
            val_check_interval=5000,
            num_sanity_val_steps=0,
            strategy="auto",
            devices=1,
            accelerator="gpu"
        )

        # --- 7. Start Training ---
        # The `ckpt_path` argument tells the trainer to resume from a checkpoint.
        # If ckpt_path is None, it starts a new training run.
        trainer.fit(
            model=model,
            train_dataloaders=dataloader,
            # val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path
        )

        wandb.finish()
    return (train_run,)


@app.cell
def _(train_run):
    def _make_sure_scratch_is_clean():
        base_temp_dir = tempfile.gettempdir()
        if not os.path.isdir(base_temp_dir) or not os.access(base_temp_dir, os.W_OK):
            raise IOError(
                f"The temporary directory '{base_temp_dir}' is not a writable directory."
            )

        scratch_dir = os.path.join(base_temp_dir, "scans")

        if not os.path.exists(scratch_dir):
            os.makedirs(scratch_dir)
            print(f"Created clean scratch directory: {scratch_dir}")
            return

        print(f"Found existing scratch directory: {scratch_dir}")
        total_size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(scratch_dir)
            for f in files
            if os.path.exists(os.path.join(root, f))
        )
        print(f"Scratch directory size: {total_size / (1024**2):.2f} MB")

        entries = sorted(os.listdir(scratch_dir))
        if entries:
            print("First few entries in scratch:")
            for name in entries[:5]:
                print("  ", name)

        try:
            shutil.rmtree(scratch_dir)
            os.makedirs(scratch_dir)
            print("Successfully cleared and recreated scratch directory.")
        except OSError as e:
            print(f"Error clearing scratch directory: {e}")
            raise

    _make_sure_scratch_is_clean()
    train_run()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
