r"""
Self-supervised learning with Generalized Recorrupted-to-Recovered (GR2R)
Training Script for FMD Dataset (Microscopy Denoising)
====================================================================================================
"""

import os
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
import matplotlib.pyplot as plt

# Use local fixed dataset utilities
from loreal_dataset_fixed import FMDDataset, get_fmdd_sequences, get_fmdd_split_from_file

# ---------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------
BASE_DIR = Path(".")
PROJECT_NAME = "denoising-poisson-fmd"
ORIGINAL_DATA_DIR =  Path("../data/FMDD")
DATA_DIR = ORIGINAL_DATA_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "ckpts" / PROJECT_NAME
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Use FMDDataset from loreal_dataset_fixed

def train_model(args):
    print(f"Starting training on {device} with {args.loss} loss...")
    
    # 1. Setup Noise and Physics
    noise_model = dinv.physics.PoissonNoise(args.noise)
    # DRUNet uses sigma as a noise level input (concatenated noise level map).
    # For Poisson noise with gain=args.noise on data in [0,1]:
    # Var[y] = x * args.noise  → sigma_eff(x=0.5) = sqrt(0.5 * args.noise) ≈ 0.044
    # The original code used sigma=args.noise (≈0.004), which is 10x too small.
    noise_model.sigma = (0.5 * args.noise) ** 0.5
    physics = dinv.physics.Denoising(noise_model=noise_model)

    # 2. Setup Datasets
    img_size = 128 # Smaller crops for faster training
    transform = transforms.Compose([
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    test_transform = transforms.Compose([
        # No ToTensor needed here, FMDDDataset returns tensors.
    ])

    img_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B', 'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R', 'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH', 'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
    
    # 2. Setup Datasets directly from PNGs (Synchronized via TXT)
    print("Loading dataset directly from PNGs (synchronized via TXT)...")
    sequences = get_fmdd_sequences(ORIGINAL_DATA_DIR, modalities=img_types)
    SPLIT_FILE = "fmdd_split.txt"
    train_seq, test_seq, visualize_indices = get_fmdd_split_from_file(sequences, SPLIT_FILE)
    
    train_dataset = FMDDataset(sequence_info=train_seq, patch_size=(img_size, img_size), 
                                mode='synthetic', gamma=1.0/args.noise, data_scale=args.data_scale,
                                transform=transform, num_frames=1)
    test_dataset = FMDDataset(sequence_info=test_seq, mode='synthetic', 
                               gamma=1.0/args.noise, data_scale=args.data_scale,
                               transform=test_transform, num_frames=1)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. Setup Model (1 channel for FMD, no pretrained weights)
    model = dinv.models.ArtifactRemoval(dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained=None, nc=[16, 32, 64, 128])).to(device)

    # 4. Setup Loss and Optimizer
    if args.loss == "gr2r_mse":
        criterion = R2RLoss(noise_model=noise_model)
        model = criterion.adapt_model(model)
    else:
        raise ValueError("Only gr2r_mse is supported in this script version")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. Training Loop
    best_psnr = 0
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
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
        num_eval = min(len(test_dataset), 50)
        current_psnr = 0
        current_val_loss = 0
        with torch.no_grad():
            for i in range(num_eval): # Eval on more images
                y_noisy, x_gt = test_dataset[i]
                y_noisy, x_gt = y_noisy.unsqueeze(0).to(device), x_gt.unsqueeze(0).to(device)
                
                # Forward pass for PSNR (no update_parameters needed for standard denoising)
                x_est = model(y_noisy, physics)
                current_psnr += PSNR()(x=x_gt, x_net=x_est).item()
                
                # Forward pass for loss (R2R needs update_parameters=True and model.training=True to update internal state)
                # We use a trick: set .training=True only on the wrapper to avoid BN corruption in the base model
                model.training = True 
                x_est_loss = model(y_noisy, physics, update_parameters=True)
                loss_val = criterion(x_est_loss, y_noisy, physics, model)
                current_val_loss += loss_val.item()
                model.training = False # Back to eval for the wrapper
        
        current_psnr /= num_eval
        current_val_loss /= num_eval
        train_losses.append(epoch_loss / len(train_dataloader))
        val_losses.append(current_val_loss)

        print(f"Epoch {epoch+1} Average PSNR: {current_psnr:.2f} dB, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pth")
            print(f"New best model saved! (PSNR: {best_psnr:.2f})")

    # 6.5 Plot Losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = RESULTS_DIR / f"loss_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")

    # 7. Save final result as TIF
    print("Saving final results to TIFF...")
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = RESULTS_DIR / f"tif_output_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in visualize_indices:
        if i >= len(test_dataset):
            continue
        y_noisy, x_gt = test_dataset[i]
        y_noisy, x_gt = y_noisy.unsqueeze(0).to(device), x_gt.unsqueeze(0).to(device)
        x_est = model(y_noisy, physics)
        
        tifffile.imwrite(str(output_dir / f"fmd_{i}_denoised.tif"), x_est.squeeze().cpu().detach().numpy().astype(np.float32))
        tifffile.imwrite(str(output_dir / f"fmd_{i}_noisy.tif"), y_noisy.squeeze().cpu().detach().numpy().astype(np.float32))
        tifffile.imwrite(str(output_dir / f"fmd_{i}_clean.tif"), x_gt.squeeze().cpu().detach().numpy().astype(np.float32))
    print(f"Finished. Check results in {output_dir}")

class Args:
    loss = "gr2r_mse"
    noise = 1/255
    epochs = 20
    batch_size = 16
    lr = 1e-4
    n_images = 12000 # Number of images for quick verification
    trial = 0
    data_scale = 1.0

if __name__ == "__main__":
    args = Args()
    train_model(args)
