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
    import boto3
    from botocore.client import Config
    from botocore.exceptions import ClientError
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import imageio
    import io
    import nibabel as nib
    from pathlib import Path
    from types import SimpleNamespace


@app.class_definition
# define the LightningModule
class RadiographyEncoder(L.LightningModule):

    class MIMHead(nn.Module):
        def __init__(self, dim, pos_max_freq, patch_size, depth, layer_dropout):
            super().__init__()
            self.pos_emb = PosEmbedding3D(dim, max_freq = pos_max_freq)
            self.mim = CrossAttender(dim = dim, depth = depth, layer_dropout=layer_dropout)
            self.to_pixels = nn.Linear(dim, patch_size*patch_size)
            self.patch_size = patch_size

        def forward(self, coords, context):
            sin, cos = self.pos_emb(coords)
            sin_unrepeated = sin[:, :, 0::2]
            cos_unrepeated = cos[:, :, 0::2]
            pos = PosEmbedding3D.interleave_two_tensors(sin_unrepeated, cos_unrepeated)

            unmasked = self.mim(pos, context=context)
            unmasked = self.to_pixels(unmasked)
            unmasked = unmasked.view(unmasked.size(0), unmasked.size(1), self.patch_size, self.patch_size)
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

    def configure_optimizers(self):
        # Use the learning_rate from hparams so it can be configured by sweeps
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


@app.class_definition
class PrismOrderingDataset(IterableDataset):
    def __init__(self, metadata, patch_shape, n_patches, n_sampled_from_same_study,
                 cache_base_dir: str, r2_bucket_name: str, r2_account_id: str,
                 r2_access_key: str, r2_secret_key: str,
                 cache_size: int = 20, max_uses_per_file: int = 50):
        super().__init__()
        self.metadata = pd.read_csv(metadata)
        self.patch_shape = patch_shape
        self.n_patches = n_patches
        self.n_sampled_from_same_study = n_sampled_from_same_study

        # Cache settings
        self.cache_base_dir = Path(cache_base_dir)
        self.cache_size = cache_size
        self.max_uses_per_file = max_uses_per_file

        # R2 settings for boto3
        self.r2_bucket_name = r2_bucket_name
        self.r2_account_id = r2_account_id
        self.r2_access_key = r2_access_key
        self.r2_secret_key =r2_secret_key

        # Worker-specific attributes, initialized in __iter__
        self.worker_id = None
        self.worker_cache_dir = None
        self.cache_state = {}  # Tracks {remote_path: {"local_path": ..., "uses": ...}}
        self.s3_client = None  # S3 client will be created per worker

    def _initialize_s3_client(self):
        """Initializes the S3 client for the worker."""
        try:
            endpoint_url = f"https://{self.r2_account_id}.r2.cloudflarestorage.com"
            # Use an unsigned configuration for public buckets, no credentials needed.
            self.s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=self.r2_access_key,
                aws_secret_access_key=self.r2_secret_key,
                region_name='auto' # or your specific region
            )
        except Exception as e:
            print(f"[Worker {self.worker_id}] Failed to initialize S3 client: {e}")
            self.s3_client = None


    def _download_file(self, remote_path, local_path):
        """Downloads a single file from R2 to a local path using boto3."""
        if not self.s3_client:
            print(f"[Worker {self.worker_id}] S3 client not initialized. Cannot download.")
            return False
        try:
            self.s3_client.download_file(
                self.r2_bucket_name,
                remote_path,
                str(local_path) # boto3 expects a string path
            )
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == '404':
                print(f"[Worker {self.worker_id}] Error: s3://{self.r2_bucket_name}/{remote_path} not found.")
            else:
                print(f"[Worker {self.worker_id}] S3 ClientError downloading {remote_path}: {e}")
            return False
        except Exception as e:
            print(f"[Worker {self.worker_id}] Unexpected error downloading {remote_path}: {e}")
            return False

    def _compute_robust_stats(self, data_array):
        """Computes median and stdev after filtering out extreme values."""
        stdev = np.std(data_array)
        values, counts = np.unique(data_array, return_counts=True)

        num_top_to_remove = int(np.maximum(stdev // 10, 1))
        if num_top_to_remove >= len(values):
            num_top_to_remove = len(values) - 1

        top_indices = np.argsort(counts)[-num_top_to_remove:]
        mask = ~np.isin(data_array, values[top_indices])

        filtered_img = data_array[mask]
        if filtered_img.size == 0:
            return np.median(data_array), np.std(data_array)

        median = np.median(filtered_img)
        stdev = np.std(filtered_img)
        return median, stdev

    def _populate_initial_cache(self, worker_metadata):
        """Fills the cache for the first time."""
        print(f"[Worker {self.worker_id}] Populating initial cache of size {self.cache_size}...")
        while len(self.cache_state) < self.cache_size:
            sample = worker_metadata[~worker_metadata['file_path'].isin(self.cache_state.keys())].sample(n=1).iloc[0]
            remote_path = sample["file_path"]
            local_filename = remote_path.replace("/", "_")
            local_path = self.worker_cache_dir / local_filename

            if self._download_file(remote_path, local_path):
                self.cache_state[remote_path] = {"local_path": local_path, "uses": 0}
        print(f"[Worker {self.worker_id}] Cache populated.")

    def _replace_cached_file(self, remote_path_to_replace, worker_metadata):
        """Deletes an old file and downloads a new one to replace it."""
        # 1. Remove old file
        file_info = self.cache_state.pop(remote_path_to_replace, None)
        if file_info:
            try:
                os.remove(file_info["local_path"])
            except OSError as e:
                print(f"[Worker {self.worker_id}] Error removing {file_info['local_path']}: {e}")

        # 2. Add a new file
        new_file_added = False
        while not new_file_added:
            sample = worker_metadata[~worker_metadata['file_path'].isin(self.cache_state.keys())].sample(n=1).iloc[0]
            new_remote_path = sample["file_path"]
            local_filename = new_remote_path.replace("/", "_")
            new_local_path = self.worker_cache_dir / local_filename

            if self._download_file(new_remote_path, new_local_path):
                self.cache_state[new_remote_path] = {"local_path": new_local_path, "uses": 0}
                new_file_added = True
                print(f"[Worker {self.worker_id}] Rotated cache: Replaced {remote_path_to_replace} with {new_remote_path}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.worker_id = worker_info.id if worker_info else 0

        if worker_info is None:
            worker_metadata = self.metadata
        else:
            worker_metadata = self.metadata.iloc[self.worker_id::worker_info.num_workers]
            seed = (torch.initial_seed() + self.worker_id) % (2**32)
            np.random.seed(seed)
            random.seed(seed)

        # --- Boto3 S3 Client Initialization (per-worker) ---
        self._initialize_s3_client()

        # --- Setup Worker Cache ---
        self.worker_cache_dir = self.cache_base_dir / f"worker_{self.worker_id}"
        shutil.rmtree(self.worker_cache_dir, ignore_errors=True)
        self.worker_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_state = {}
        self._populate_initial_cache(worker_metadata)

        while True:
            if not self.cache_state:
                print(f"[Worker {self.worker_id}] Cache is empty, cannot proceed.")
                break

            remote_path = random.choice(list(self.cache_state.keys()))
            file_info = self.cache_state[remote_path]
            local_path = file_info["local_path"]

            try:
                nifti_image = nib.as_closest_canonical(nib.load(local_path))
                image_data = nifti_image.get_fdata()
                median, stdev = self._compute_robust_stats(image_data)

                # Assuming nifti_scan is defined
                scan = nifti_scan(
                    path_to_scan=str(local_path),
                    median=median,
                    stdev=stdev,
                    base_patch_size=self.patch_shape
                )

                for _ in range(self.n_sampled_from_same_study):
                    yield self.generate_training_pair(scan, n_patches=self.n_patches)

                file_info["uses"] += 1
                if file_info["uses"] >= self.max_uses_per_file:
                    self._replace_cached_file(remote_path, worker_metadata)

            except Exception as e:
                print(f"[Worker {self.worker_id}] CRITICAL: Error processing {remote_path}, replacing it. Error: {e}")
                self._replace_cached_file(remote_path, worker_metadata)
                continue

    # `generate_training_pair` method remains unchanged from your original code
    def generate_training_pair(self, scan, n_patches: int, debug: bool = False) -> tuple:
        # This implementation is a placeholder based on your original code
        # You would use your actual `scan.get_random_wc_ww_for_scan` and `scan.train_sample` here
        wc1, ww1 = scan.get_random_wc_ww_for_scan()
        wc2, ww2 = scan.get_random_wc_ww_for_scan()
        sample_1_data = scan.train_sample(n_patches=n_patches, wc=wc1, ww=ww1)
        sample_2_data = scan.train_sample(n_patches=n_patches, wc=wc2, ww=ww2)
        patches_1 = sample_1_data['normalized_patches']
        patches_2 = sample_2_data['normalized_patches']
        patch_coords_1 = sample_1_data['relative_patch_centers_pt']
        patch_coords_2 = sample_2_data['relative_patch_centers_pt']
        pos_label = (sample_2_data['prism_center_pt'] - sample_1_data['prism_center_pt'])
        rotation_label = (np.array(sample_2_data['rotation_degrees']) - np.array(sample_1_data['rotation_degrees']))
        window_label = np.array([wc2 - wc1, ww2 - ww1])
        label = np.concatenate((pos_label, rotation_label, window_label))
        tensors = [patches_1, patches_2, patch_coords_1, patch_coords_2, label]
        if debug:
            return tuple(torch.from_numpy(arr).to(torch.float32) for arr in tensors), sample_1_data, sample_2_data
        return tuple(torch.from_numpy(arr).to(torch.float32) for arr in tensors)


@app.cell
def _():
    def get_allocated_cpus():
        return int(os.environ.get("SLURM_CPUS_PER_GPU", 8))

    def get_gpu_memory_gb():
        gpu_memory = {}
        if not torch.cuda.is_available():
            return {0: 24.0} # Fallback for local testing without GPU
        for i in range(torch.cuda.device_count()):
            total_mem_bytes = torch.cuda.get_device_properties(i).total_memory
            total_mem_gb = round(total_mem_bytes / (1024**3), 2)
            gpu_memory[i] = total_mem_gb
        return gpu_memory

    def train_run():
        """
        Main training function adapted for the new caching dataset.
        """
        run = wandb.init(resume="allow")
        print(f"Run config:\n{run.config}")
        print("="*20)

        checkpoint_dir = f'../checkpoints/{run.id}'
        cfg = run.config

        # --- Resuming logic remains the same ---
        ckpt_path = None
        if run.resumed:
            # ... (Your existing resume logic) ...
            # This part does not need to be changed.
            pass

        # --- 3. Setup Model (Unchanged) ---
        model = RadiographyEncoder(
            encoder_dim=cfg.encoder_dim,
            encoder_depth=cfg.encoder_depth,
            encoder_heads=cfg.encoder_heads,
            mlp_dim=cfg.mlp_dim,
            n_registers=cfg.n_registers,
            pos_max_freq=cfg.pos_max_freq,
            use_rotary=cfg.use_rotary,
            use_absolute=cfg.use_absolute,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            patch_size=cfg.patch_size,
            pos_objective_mode=cfg.pos_objective_mode,
            window_objective=cfg.window_objective,
            scan_contrastive_objective=cfg.scan_contrastive_objective,
            mim_objective=cfg.mim_objective,
        )

        # --- 4. Setup Data ---
        # NOTE: Your new PrismOrderingDataset expects a `file_path` column in the CSV.
        # Ensure '300_labeled.csv' has this column.
        METADATA_PATH = "300_labeled.csv"
        N_PATCHES = 64
        NUM_WORKERS = int(get_allocated_cpus()) - 2

        CACHE_DIR = tempfile.gettempdir()

        if not os.path.isdir(CACHE_DIR) or not os.access(CACHE_DIR, os.W_OK):
            raise IOError(
                f"The determined temporary directory '{CACHE_DIR}' "
                "is not a writable directory. Check system configuration and permissions."
            )

        CACHE_DIR = os.path.join(CACHE_DIR, "scans")
        os.makedirs(CACHE_DIR, exist_ok=True)

        # --- UPDATED DATASET INSTANTIATION ---
        dataset = PrismOrderingDataset(
            metadata=METADATA_PATH,
            patch_shape=cfg.patch_size,
            n_patches=N_PATCHES,
            n_sampled_from_same_study=cfg.num_repeated_study_samples,
            # New cache arguments
            cache_base_dir=CACHE_DIR,
            cache_size=20,
            max_uses_per_file=50,
            ###
        )

        dataloader = DataLoader(
            dataset,
            batch_size=int(cfg.batch_size * (get_gpu_memory_gb()[0] / 100.0)),
            num_workers=int(NUM_WORKERS),
            persistent_workers=(NUM_WORKERS > 0),
            pin_memory=True,
        )

        wandb_logger = WandbLogger(log_model="all")

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{step}',
            every_n_train_steps=2000,
            save_last=True,
        )

        # --- 6. Setup Trainer (Unchanged) ---
        trainer = L.Trainer(
            max_epochs=-1,
            max_steps=5000000,
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=int(80.0 / get_gpu_memory_gb()[0]),
            logger=wandb_logger,
            log_every_n_steps=25,
            val_check_interval=5000,
            num_sanity_val_steps=0,
            strategy="auto",
            devices=-1,
            accelerator="gpu"
        )

        # --- 7. Start Training (Unchanged) ---
        trainer.fit(
            model=model,
            train_dataloaders=dataloader,
            # val_dataloaders=val_dataloader, # Validation logic can be added later
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
