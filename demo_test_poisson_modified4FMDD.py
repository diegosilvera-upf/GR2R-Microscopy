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
import matplotlib.pyplot as plt

# Use local fixed dataset utilities
from loreal_dataset_fixed import FMDDataset, get_fmdd_sequences, get_fmdd_split_from_file

# ---------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------
BASE_DIR = Path(".")
PROJECT_NAME = "denoising-poisson-fmdd-drunet"
ORIGINAL_DATA_DIR =  Path("../data/FMDD")
RESULTS_DIR = BASE_DIR / "results" / PROJECT_NAME
CKPT_DIR = BASE_DIR / "ckpts" / PROJECT_NAME
CKPT_DIR.mkdir(parents=True, exist_ok=True) #parents=True crea los directorios "padres" si no existen. exist_ok=True evita que de error si la carpeta ya existe.
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
    noise_model = dinv.physics.PoissonNoise(args.noise, normalize=True) #normalize=True para multiplicar por la ganancia
    # DRUNet uses sigma as a noise level input (concatenated noise level map).
    # For Poisson noise with gain=args.noise on data in [0,1]:
    # Var[y] = x * args.noise  → sigma_eff(x=0.5) = sqrt(0.5 * args.noise) ≈ 0.044
    # The original code used sigma=args.noise (≈0.004), which is 10x too small.
    noise_model.sigma = (0.5 * args.noise) ** 0.5
    physics = dinv.physics.Denoising(noise_model=noise_model)

    # 2. Setup Datasets
    # The name "img_types" is confusing. These are the names of the folders containing the sequences
    img_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B', 'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R', 'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH', 'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
    
    # Discover FMDD sequences
    sequences = get_fmdd_sequences(ORIGINAL_DATA_DIR, modalities=img_types)
    print(f"Found {len(sequences)} sequences.")

    # Split into train/test using explicit TXT file
    SPLIT_FILE = "fmdd_split.txt"
    train_seq, test_seq, visualize_indices = get_fmdd_split_from_file(sequences, SPLIT_FILE)
    print(f"Split loaded from {SPLIT_FILE}: {len(train_seq)} train, {len(test_seq)} test sequences.")

    # Data Augmentation (D4 symmetry group). TRANSFORMADAS DE TORCH. NO SE ESTA USANDO EL TRANSFORM DE LOREAL
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Add rotation if needed, but flips are often enough for 2D
    ])
    
    # num_frames=1 for 2D DRUNet
    train_dataset = FMDDataset(sequence_info=train_seq, patch_size=(args.patch_size, args.patch_size), 
                                mode='synthetic', gamma=1.0/args.noise, data_scale=args.data_scale, 
                                num_frames=1, transform=transform) #gamma=1.0/args.noise??? POSIBLE ERROR
    test_dataset = FMDDataset(sequence_info=test_seq, mode='synthetic', 
                               gamma=1.0/args.noise, data_scale=args.data_scale, 
                               num_frames=1) #gamma=1.0/args.noise??? POSIBLE ERROR

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
    train_losses = []
    val_losses = []

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
        current_val_loss = 0
        # visualize_indices is loaded from the TXT split file
        n_eval = min(len(test_dataset), 10)
        with torch.no_grad():
            for i in range(n_eval):
                y_stack, x_gt = test_dataset[i]
                y_stack, x_gt = y_stack.unsqueeze(0).to(device), x_gt.unsqueeze(0).to(device)
                
                # y_stack[:, 0:1, :, :] is the only frame
                y_noisy = y_stack[:, 0:1, :, :].clone()
                
                # Forward pass for PSNR
                x_est = model(y_noisy, physics)
                current_psnr += PSNR()(x=x_gt, x_net=x_est).item()
                
                # Forward pass for loss (R2R needs update_parameters=True and model.training=True to update internal state)
                # We use a trick: set .training=True only on the wrapper to avoid BN corruption in the base model
                model.training = True 
                x_est_loss = model(y_noisy, physics, update_parameters=True)
                loss_val = criterion(x_est_loss, y_noisy, physics, model)
                current_val_loss += loss_val.item()
                model.training = False # Back to eval for the wrapper
        
        current_psnr /= n_eval
        current_val_loss /= n_eval
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
    loss_plot_path = output_dir / f"loss_plot.png"
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")

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
    epochs = 20
    batch_size = 32 # Normalized batch size
    lr = 1e-4
    patch_size = 256
    data_scale = 1.0 # Dataset now handles normalization to [0, 1]

if __name__ == "__main__":
    args = Args()
    train_model(args)
