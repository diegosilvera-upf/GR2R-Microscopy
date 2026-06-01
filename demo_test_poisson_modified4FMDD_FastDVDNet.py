r"""
Self-supervised learning with Generalized Recorrupted-to-Recovered (GR2R)
Training Script for FMD Dataset (Microscopy Denoising) using FastDVDNet
====================================================================================================
"""

import os
import sys
from datetime import datetime
import deepinv as dinv
from torch.utils.data import DataLoader
import torchvision
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from torchvision import transforms
from deepinv.loss import PSNR, SSIM, R2RLoss
import tifffile
from tqdm import tqdm

# Use local fixed dataset utilities
from loreal_dataset_fixed import FMDDDataset, get_fmdd_sequences, get_fmdd_split_from_file
from models_FastDVDnet_sans_noise_map import FastDVDnet

class FastDVDnetR2RWrapper(torch.nn.Module):
    """
    Wrapper to make FastDVDnet compatible with deepinv's R2RLoss.
    It takes a 5-frame stack as context and allows R2RLoss to perturb the central frame.
    """
    def __init__(self, model, alpha=0.15):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self._context = None

    def set_context(self, stack):
        """Stores the 5-frame stack before the forward pass."""
        self._context = stack.detach()

    def forward(self, y_central, physics=None, update_parameters=False, **kwargs):
        if self._context is None:
            raise RuntimeError("Call set_context(stack) before forward pass.")
        
        # Clone to avoid modifying the original stack
        stack = self._context.clone()
        
        # SNR Consistency: Recorrupt the rest of the stack to match y_central's noise level
        if self.training:
            with torch.no_grad():
                gain = physics.noise_model.gain if (physics is not None and hasattr(physics.noise_model, 'gain')) else args.noise
                for i in [0, 1, 3, 4]:
                    y_neighbor = stack[:, i:i+1, :, :]
                    z = y_neighbor / gain
                    # alpha is the probability of removal in deepinv's set_binomial_corruptor
                    sampler = torch.distributions.Binomial(torch.clamp(torch.round(z), min=0), self.alpha)
                    stack[:, i:i+1, :, :] = gain * (z - sampler.sample()) / (1.0 - self.alpha)
        
        # Replace central frame with the (already recorrupted) y_central
        stack[:, 2:3, :, :] = y_central
        
        # FastDVDnet handles the forward pass
        return self.model(stack)

# ---------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------
BASE_DIR = Path(".")
PROJECT_NAME = "denoising-poisson-fmdd-fastdvdnet"
ORIGINAL_DATA_DIR =  Path("../data/FMDD")
# DATA_DIR = ORIGINAL_DATA_DIR / "measurements" #No lo usa?
RESULTS_DIR = BASE_DIR / "results"/ PROJECT_NAME
CKPT_DIR = BASE_DIR / "ckpts" / PROJECT_NAME
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

def save_parameters(args, output_dir):
    """Save experiment parameters to a text file in the output directory."""
    with open(output_dir / "parameters.txt", "w") as f:
        f.write(f"Experiment: demo_test_poisson_modified4FMDD_FastDVDNet.py\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write("-" * 50 + "\n")
        # Capture all non-private, non-callable attributes (class and instance)
        for key in dir(args):
            if not key.startswith("_"):
                value = getattr(args, key)
                if not callable(value):
                    f.write(f"{key} = {value}\n")
    print(f"Parameters saved to {output_dir / 'parameters.txt'}")

def train_model(args):

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = RESULTS_DIR / f"tif_output_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_parameters(args, output_dir)

    print(f"Starting training on {device} with {args.loss} loss...")
    
    # 1. Setup Noise and Physics
    noise_model = dinv.physics.PoissonNoise(args.noise)
    physics = dinv.physics.Denoising(noise_model=noise_model)

    # 2. Setup Datasets
    img_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B', 'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R', 'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH', 'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
    
    # Discover FMDD sequences
    sequences = get_fmdd_sequences(ORIGINAL_DATA_DIR, modalities=img_types)
    print(f"Found {len(sequences)} sequences.")

    # Split into train/test using explicit TXT file
    SPLIT_FILE = "fmdd_split.txt"
    # visualize_indices is loaded from the TXT split file
    train_seq, test_seq, visualize_indices = get_fmdd_split_from_file(sequences, SPLIT_FILE)
    print(f"Split loaded from {SPLIT_FILE}: {len(train_seq)} train, {len(test_seq)} test sequences.")

    # Data Augmentation (D4 symmetry group)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    # Use FMDDDataset in synthetic mode (creates stacks from GT)
    patch_size = 128
    data_scale = 1.0 # Dataset now handles normalization to [0, 1]
    
    train_dataset = FMDDDataset(sequence_info=train_seq, patch_size=(patch_size, patch_size), 
                                mode='synthetic', gamma=1.0/args.noise, data_scale=data_scale,
                                transform=transform)
    test_dataset = FMDDDataset(sequence_info=test_seq, mode='synthetic', 
                               gamma=1.0/args.noise, data_scale=data_scale)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 3. Setup Model (FastDVDnet)
    base_model = FastDVDnet(num_input_frames=5).to(device)
    model = FastDVDnetR2RWrapper(base_model, alpha=args.alpha).to(device)

    # 4. Setup Loss and Optimizer
    if args.loss == "gr2r_mse":
        criterion = R2RLoss(noise_model=noise_model, alpha=args.alpha)
        model = criterion.adapt_model(model)
    else:
        raise ValueError("Only gr2r_mse is supported in this script version")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. Training Loop
    best_psnr = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for stack, y in pbar:
            stack, y = stack.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Set context for video R2R
            if hasattr(model, "model") and hasattr(model.model, "set_context"):
                model.model.set_context(stack)
            else:
                model.set_context(stack)
            
            # Forward pass with update_parameters=True to store corruption for R2R
            x_est = model(y, physics, update_parameters=True)
            
            # Compute loss
            loss = criterion(x_est, y, physics, model)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        # 6. Evaluation
        model.eval()
        current_psnr = 0
        n_eval = min(len(test_dataset), 10)
        with torch.no_grad():
            for i in range(n_eval):
                stack, x_gt = test_dataset[i]
                stack, x_gt = stack.unsqueeze(0).to(device), x_gt.unsqueeze(0).to(device)
                
                # Use central frame of stack as input y
                y_noisy = stack[:, 2:3, :, :].clone()
                
                # Use inner model for clean inference
                inner_model = model.model if hasattr(model, "model") else model
                inner_model.set_context(stack)
                x_est = inner_model(y_noisy, physics)
                
                current_psnr += PSNR()(x=x_gt, x_net=x_est).item()
        
        current_psnr /= n_eval
        print(f"Epoch {epoch+1} Average PSNR: {current_psnr:.2f} dB")

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pth")
            print(f"New best model saved! (PSNR: {best_psnr:.2f})")

    # 7. Save final result as TIF
    print("Saving final results to TIFF...")
    model.eval()
    inner_model = model.model if hasattr(model, "model") else model
    with torch.no_grad():
        for i in [0, 1, 2]: # Samples from test set
            if i >= len(test_dataset): break
            stack, x_gt = test_dataset[i]
            stack, x_gt = stack.unsqueeze(0).to(device), x_gt.unsqueeze(0).to(device)
            y_noisy = stack[:, 2:3, :, :].clone()
            
            inner_model.set_context(stack)
            x_est = inner_model(y_noisy, physics)
            
            tifffile.imwrite(str(output_dir / f"fmd_{i}_denoised.tif"), x_est.squeeze().cpu().numpy().astype(np.float32))
            tifffile.imwrite(str(output_dir / f"fmd_{i}_noisy.tif"), y_noisy.squeeze().cpu().numpy().astype(np.float32))
            tifffile.imwrite(str(output_dir / f"fmd_{i}_clean.tif"), x_gt.squeeze().cpu().numpy().astype(np.float32))
    
    print(f"Finished. Check results in {output_dir}")

class Args:
    loss = "gr2r_mse"
    noise = 1/255.0
    alpha = 0.15
    epochs = 200
    batch_size = 16 # Reduced for stack-based training
    lr = 1e-4

if __name__ == "__main__":
    args = Args()
    train_model(args)
