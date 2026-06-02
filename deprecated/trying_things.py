r"""
Self-supervised learning with Generalized Recorrupted-to-Recovered (GR2R)
Training Script for FMD Dataset (Microscopy Denoising) using DRUNet
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

# Add Loreal directory to sys.path to import dataset utilities
sys.path.append(str(Path("../Loreal").absolute()))
from dataset import FMDDDataset, get_fmdd_sequences

# ---------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------
BASE_DIR = Path(".")
PROJECT_NAME = "denoising-poisson-fmdd-drunet"
ORIGINAL_DATA_DIR =  Path("../data/FMDD")
RESULTS_DIR = BASE_DIR / "results" / PROJECT_NAME
CKPT_DIR = BASE_DIR / "ckpts" / PROJECT_NAME
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

def save_parameters(args, output_dir):
    """Save experiment parameters to a text file in the output directory."""
    with open(output_dir / "parameters.txt", "w") as f:
        f.write(f"Experiment: demo_test_poisson_modified4FMDD.py\n")
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
    noise_model.sigma = args.noise # Crucial for DRUNet to find the noise level
    physics = dinv.physics.Denoising(noise_model=noise_model)

    # 2. Setup Datasets
    img_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B', 'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R', 'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH', 'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
    
    # Discover FMDD sequences
    sequences = get_fmdd_sequences(ORIGINAL_DATA_DIR, modalities=img_types)
    print(f"Found {len(sequences)} sequences.")

    # Split into train/test (Same logic as FastDVDNet)
    n_total = len(sequences)
    n_train = int(11 * n_total / 12) 
    train_seq = sequences[:n_train]
    test_seq = sequences[n_train:]

    # Data Augmentation (D4 symmetry group)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Add rotation if needed, but flips are often enough for 2D
    ])

    patch_size = 128
    data_scale = 255.0 
    
    # num_frames=1 for 2D DRUNet
    train_dataset = FMDDDataset(sequence_info=train_seq, patch_size=(patch_size, patch_size), 
                                mode='synthetic', gamma=1.0/args.noise, data_scale=data_scale, 
                                num_frames=1, transform=transform)
    test_dataset = FMDDDataset(sequence_info=test_seq, mode='synthetic', 
                               gamma=1.0/args.noise, data_scale=data_scale, 
                               num_frames=1)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 3. Setup Model (DRUNet)
    model = dinv.models.ArtifactRemoval(dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained=None, nc=[16, 32, 64, 128])).to(device)

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
        
        for y_stack, y in pbar:
            # y_stack has shape [B, 1, H, W] for DRUNet (num_frames=1)
            y_stack, y = y_stack.to(device), y.to(device)
            optimizer.zero_grad()
            
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
                y_stack, x_gt = test_dataset[i]
                y_stack, x_gt = y_stack.unsqueeze(0).to(device), x_gt.unsqueeze(0).to(device)
                
                # y_stack[:, 0:1, :, :] is the only frame
                y_noisy = y_stack[:, 0:1, :, :].clone()
                
                x_est = model(y_noisy, physics)
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
    with torch.no_grad():
        for i in [0, 1, 2]: # Samples from test set
            if i >= len(test_dataset): break
            y_stack, x_gt = test_dataset[i]
            y_stack, x_gt = y_stack.unsqueeze(0).to(device), x_gt.unsqueeze(0).to(device)
            y_noisy = y_stack[:, 0:1, :, :].clone()
            
            x_est = model(y_noisy, physics)
            
            tifffile.imwrite(str(output_dir / f"fmd_{i}_denoised.tif"), x_est.squeeze().cpu().numpy().astype(np.float32))
            tifffile.imwrite(str(output_dir / f"fmd_{i}_noisy.tif"), y_noisy.squeeze().cpu().numpy().astype(np.float32))
            tifffile.imwrite(str(output_dir / f"fmd_{i}_clean.tif"), x_gt.squeeze().cpu().numpy().astype(np.float32))
    
    print(f"Finished. Check results in {output_dir}")

class Args:
    loss = "gr2r_mse"
    noise = 1/255.0
    alpha = 0.15
    epochs = 50
    batch_size = 16 # Normalized batch size
    lr = 1e-4

if __name__ == "__main__":
    args = Args()
    train_model(args)
