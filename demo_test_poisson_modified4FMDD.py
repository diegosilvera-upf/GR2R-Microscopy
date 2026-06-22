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
from training_utils import save_parameters

# Use local fixed dataset utilities
# from loreal_dataset_fixed import FMDDataset, get_fmdd_sequences, get_fmdd_split_from_file
from dataset import FMDDataset, get_fmdd_sequences, get_fmdd_split_from_file

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

def build_eval_cache(test_dataset, physics, n_eval, device, seed):
    """
    Pre-generate fixed (x_gt, y_noisy) pairs.

    Without this, Poisson noise is resampled every epoch and PSNR/val_loss are not
    comparable across epochs, which makes the model look like it improves every time.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cache = []
    with torch.no_grad():
        for i in range(n_eval):
            x_clean_stack, x_clean = test_dataset[i]
            x_clean_stack = x_clean_stack.unsqueeze(0).to(device)
            x_gt = x_clean.unsqueeze(0).to(device)
            x_clean_img = x_clean_stack[:, 0:1, :, :]
            y_noisy = physics(x_clean_img) #Acá es donde agrego ruido sintético
            cache.append((x_gt.detach().cpu(), y_noisy.detach().cpu()))

    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    print(f"Built eval cache with {len(cache)} fixed noisy samples (seed={seed}).")
    return cache


def evaluate_epoch(model, criterion, physics, eval_cache, device, eval_seed):
    """Evaluate on fixed noisy inputs; R2R loss uses a fixed seed per sample."""
    model.eval()
    psnr_sum = 0.0
    val_loss_sum = 0.0
    n_eval = len(eval_cache)

    with torch.no_grad():
        for i, (x_gt_cpu, y_noisy_cpu) in enumerate(eval_cache):
            x_gt = x_gt_cpu.to(device)
            y_noisy = y_noisy_cpu.to(device)

            x_est = model(y_noisy, physics)
            psnr_sum += PSNR()(x=x_gt, x_net=x_est).item()

            torch.manual_seed(eval_seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_seed + i)
            model.training = True
            x_est_loss = model(y_noisy, physics, update_parameters=True)
            val_loss_sum += criterion(x_est_loss, y_noisy, physics, model).item()
            model.training = False

    return psnr_sum / n_eval, val_loss_sum / n_eval


def train_model(args):
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = RESULTS_DIR / f"tif_output_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_parameters(args, output_dir, script_name=Path(__file__).name, device=device)

    print(f"Starting training on {device} with {args.loss} loss...")
    
    # 1. Setup Noise and Physics
    noise_model = dinv.physics.PoissonNoise(args.gamma)
    noise_model.sigma = args.gamma # Robado del notebook, no lo entiendo la verdad. 
    physics = dinv.physics.Denoising(noise_model=noise_model)

    # 2. Setup Datasets
    # The name "img_types" is confusing. These are the names of the folders containing the sequences
    img_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B', 'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R', 'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH', 'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
    
    # Discover FMDD sequences
    sequences = get_fmdd_sequences(ORIGINAL_DATA_DIR, modalities=img_types)
    print(f"Found {len(sequences)} sequences.")

    # Split into train/test using explicit TXT file
    SPLIT_FILE = "txts/fmdd_split.txt"
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
                                mode='clean', data_scale=args.data_scale, 
                                num_frames=1, transform=transform, repeats_per_sequence=55)
    test_dataset = FMDDataset(sequence_info=test_seq, mode='clean', 
                               data_scale=args.data_scale, 
                               num_frames=1, repeats_per_sequence=1)

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

    n_eval = len(test_seq)
    if args.n_eval_sequences is not None:
        n_eval = min(len(test_seq), args.n_eval_sequences)
    eval_cache = build_eval_cache(test_dataset, physics, n_eval, device, args.eval_seed)
    
    # 5. Training Loop — checkpoint by val_loss on fixed eval data (not noisy PSNR)
    best_val_loss = float("inf")
    best_psnr = 0.0
    best_epoch = -1
    best_ckpt_path = output_dir / "best_model.pth"
    train_losses = []
    val_losses = []
    val_psnrs = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for x_clean_stack, x_clean in pbar:
            # x_clean_stack/x_clean tienen forma [B, 1, H, W] — imágenes limpias del GT
            x_clean_stack, x_clean = x_clean_stack.to(device), x_clean.to(device)
            # x_clean_stack[:, 0:1] es el único frame limpio (num_frames=1)
            x_clean_img = x_clean_stack[:, 0:1, :, :]
            optimizer.zero_grad()
            
            # Aplicar ruido Poisson via physics: y ~ Poisson(x_clean * gamma) / gamma
            y = physics(x_clean_img)
            
            # Forward pass con update_parameters=True para que R2R almacene la corrupción
            x_est = model(y, physics, update_parameters=True)
            
            # Compute loss
            loss = criterion(x_est, y, physics, model)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()}) #Es la barrita que muestra la loss en tiempo real

        # 6. Evaluation (fixed noisy inputs — comparable across epochs)
        current_psnr, current_val_loss = evaluate_epoch(
            model, criterion, physics, eval_cache, device, args.eval_seed
        )
        train_losses.append(epoch_loss / len(train_dataloader))
        val_losses.append(current_val_loss)
        val_psnrs.append(current_psnr)

        print(
            f"Epoch {epoch+1} Val PSNR: {current_psnr:.2f} dB, "
            f"Train Loss: {train_losses[-1]:.6f}, Val Loss: {current_val_loss:.6f}"
        )

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_psnr = current_psnr
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_ckpt_path)
            print(
                f"New best model saved to {best_ckpt_path} "
                f"(epoch={best_epoch}, val_loss={best_val_loss:.6f}, val_psnr={best_psnr:.2f} dB)"
            )
        else:
            print(
                f"No checkpoint update (best epoch={best_epoch}, "
                f"val_loss={best_val_loss:.6f}, val_psnr={best_psnr:.2f} dB)"
            )

    with open(output_dir / "best_checkpoint.txt", "w") as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_loss={best_val_loss:.6f}\n")
        f.write(f"best_val_psnr={best_psnr:.4f}\n")
        f.write(f"weights_path={best_ckpt_path}\n")

    # 6.5 Plot Losses
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    ax1.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    if len(val_psnrs) == args.epochs:
        ax2 = ax1.twinx()
        ax2.plot(range(1, args.epochs + 1), val_psnrs, "g--", label="Val PSNR (dB)")
        ax2.set_ylabel("Val PSNR (dB)")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
    else:
        ax1.legend()
    ax1.set_title("Training and Validation Loss")
    ax1.grid(True)
    loss_plot_path = output_dir / f"loss_plot.png"
    fig.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")

    # 7. Save final result as TIF (using best-epoch weights)
    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"Loaded best weights from epoch {best_epoch} for export.")
    print("Saving final results to TIFF...")
    model.eval()
    with torch.no_grad():
        # Prioritize explicit visualization selections from fmdd_split.txt.
        # Fallback to the first 3 test samples if no visualize flag is provided.
        export_indices = visualize_indices if len(visualize_indices) > 0 else [0, 1, 2]
        for i in export_indices:
            if i >= len(test_dataset): break
            x_clean_stack, x_clean = test_dataset[i]
            x_clean_stack = x_clean_stack.unsqueeze(0).to(device)
            x_gt = x_clean.unsqueeze(0).to(device)
            x_clean_img = x_clean_stack[:, 0:1, :, :]
            torch.manual_seed(args.eval_seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.eval_seed + i)
            y_noisy = physics(x_clean_img)
            
            x_est = model(y_noisy, physics)
            
            tifffile.imwrite(str(output_dir / f"fmd_{i}_denoised.tif"), x_est.squeeze().cpu().numpy().astype(np.float32))
            tifffile.imwrite(str(output_dir / f"fmd_{i}_noisy.tif"), y_noisy.squeeze().cpu().numpy().astype(np.float32))
            tifffile.imwrite(str(output_dir / f"fmd_{i}_clean.tif"), x_gt.squeeze().cpu().numpy().astype(np.float32))
    
    print(f"Finished. Check results in {output_dir}")

class Args:
    loss = "gr2r_mse"
    gamma = 1/255.0
    alpha = 0.15
    epochs = 3
    batch_size = 16 # Normalized batch size
    lr = 1e-4
    patch_size = 256
    data_scale = 1.0 # Dataset now handles normalization to [0, 1]
    n_eval_sequences = None  # None = all test sequences; set to int to cap (e.g. 10 for faster epochs)
    eval_seed = 42  # fixed noise + R2R corruption for comparable validation

if __name__ == "__main__":
    args = Args()
    train_model(args)
