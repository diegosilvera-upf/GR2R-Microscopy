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
from matplotlib import pyplot as plt

# Use local fixed dataset utilities
from loreal_dataset_fixed import FMDDataset, get_fmdd_sequences, get_fmdd_split_from_file
from models_FastDVDnet_sans_noise_map import FastDVDnet

# class FastDVDnetR2RWrapper(torch.nn.Module):
#     """
#     Wrapper to make FastDVDnet compatible with deepinv's R2RLoss.
#     It takes a 5-frame stack as context and allows R2RLoss to perturb the central frame.
#     """
#     def __init__(self, model, alpha=0.15):
#         super().__init__()
#         self.model = model
#         self.alpha = alpha
#         self._context = None

#     def set_context(self, stack):
#         """Stores the 5-frame stack before the forward pass."""
#         self._context = stack.detach()

#     def forward(self, y_central, physics=None, update_parameters=False, **kwargs):
#         if self._context is None:
#             raise RuntimeError("Call set_context(stack) before forward pass.")
        
#         # Clone to avoid modifying the original stack
#         stack = self._context.clone()
        
#         # SNR Consistency: Recorrupt the rest of the stack to match y_central's noise level
#         if self.training:
#             with torch.no_grad():
#                 gain = physics.noise_model.gain if (physics is not None and hasattr(physics.noise_model, 'gain')) else args.noise
#                 for i in [0, 1, 3, 4]:
#                     y_neighbor = stack[:, i:i+1, :, :]
#                     z = y_neighbor / gain
#                     # alpha is the probability of removal in deepinv's set_binomial_corruptor
#                     sampler = torch.distributions.Binomial(torch.clamp(torch.round(z), min=0), self.alpha)
#                     stack[:, i:i+1, :, :] = gain * (z - sampler.sample()) / (1.0 - self.alpha)
        
#         # Replace central frame with the (already recorrupted) y_central
#         stack[:, 2:3, :, :] = y_central
        
#         # FastDVDnet handles the forward pass
#         return self.model(stack)

#I'll try the easiest context wrapper, the other one seems overcomplicated
class FastDVDNetContextWrapper(torch.nn.Module):
    """Wraps FastDVDnet to manage the 5-frame context for R2RLoss.
    Only the central frame (position 2) is replaced by R2R's recorrupted input.
    Context frames stay at their original noise level — intentional: better temporal context."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._context = None

    def set_context(self, stack):
        self._context = stack.detach()

    def forward(self, y_central, physics=None, **kwargs):
        if self._context is None:
            raise RuntimeError("Call set_context(stack) before forward pass.")
        stack = self._context.clone()
        stack[:, 2:3, :, :] = y_central
        return self.model(stack)

# ---------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------
BASE_DIR = Path(".")
PROJECT_NAME = "denoising-poisson-fmdd-fastdvdnet-retry"
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

#This function is to eval as in demo_test_poisson_modified4FMDD.py
def build_eval_cache(test_dataset, physics, n_eval, device, seed):
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
            noisy_frames = [physics(x_clean_stack[:, j:j+1]) for j in range(5)]
            stack_noisy = torch.cat(noisy_frames, dim=1)  # (1, 5, H, W)
            cache.append((x_gt.detach().cpu(), stack_noisy.detach().cpu()))

    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    print(f"Built eval cache with {len(cache)} fixed noisy samples (seed={seed}).")
    return cache

#This function is to eval as in demo_test_poisson_modified4FMDD.py
def evaluate_epoch(model, criterion, physics, eval_cache, device, eval_seed):
    model.eval()
    psnr_sum = 0.0
    val_loss_sum = 0.0
    n_eval = len(eval_cache)

    with torch.no_grad():
        for i, (x_gt_cpu, stack_noisy_cpu) in enumerate(eval_cache):
            x_gt = x_gt_cpu.to(device)
            stack_noisy = stack_noisy_cpu.to(device)
            y_central = stack_noisy[:, 2:3, :, :]

            model.model.set_context(stack_noisy)
            x_est = model(y_central, physics)
            psnr_sum += PSNR()(x=x_gt, x_net=x_est).item()

            torch.manual_seed(eval_seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_seed + i)
            model.model.set_context(stack_noisy)
            model.training = True
            x_est_loss = model(y_central, physics, update_parameters=True)
            val_loss_sum += criterion(x_est_loss, y_central, physics, model).item()
            model.training = False

    return psnr_sum / n_eval, val_loss_sum / n_eval

def train_model(args):

    # timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # output_dir = RESULTS_DIR / f"tif_output_{timestamp}"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # save_parameters(args, output_dir)

    if args.inference_dir is not None:
        output_dir = Path(args.inference_dir)
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        output_dir = RESULTS_DIR / f"tif_output_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_parameters(args, output_dir)
    print(f"Starting training on {device} with {args.loss} loss...")
    
    # 1. Setup Noise and Physics
    # noise_model = dinv.physics.PoissonNoise(args.noise)
    noise_model = dinv.physics.PoissonNoise(args.gamma)
    noise_model.sigma = args.gamma
    physics = dinv.physics.Denoising(noise_model=noise_model)

    # 2. Setup Datasets
    img_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B', 'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R', 'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH', 'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
    
    # Discover FMDD sequences
    sequences = get_fmdd_sequences(ORIGINAL_DATA_DIR, modalities=img_types)
    print(f"Found {len(sequences)} sequences.")

    # Split into train/test using explicit TXT file
    SPLIT_FILE = "txts/fmdd_split.txt"
    # visualize_indices is loaded from the TXT split file
    train_seq, test_seq, visualize_indices = get_fmdd_split_from_file(sequences, SPLIT_FILE)
    print(f"Split loaded from {SPLIT_FILE}: {len(train_seq)} train, {len(test_seq)} test sequences.")

    # Data Augmentation (D4 symmetry group)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    # Use FMDDDataset in synthetic mode (creates stacks from GT)
    
    # train_dataset = FMDDDataset(sequence_info=train_seq, patch_size=(patch_size, patch_size), 
    #                             mode='synthetic', gamma=1.0/args.noise, data_scale=data_scale,
    #                             transform=transform)
    # test_dataset = FMDDDataset(sequence_info=test_seq, mode='synthetic', 
    #                            gamma=1.0/args.noise, data_scale=data_scale)

    train_dataset = FMDDataset(sequence_info=train_seq, patch_size=(args.patch_size, args.patch_size),
                            mode='clean', data_scale=args.data_scale,
                            num_frames=5, transform=transform, repeats_per_sequence=55)
    test_dataset = FMDDataset(sequence_info=test_seq, mode='clean',
                            data_scale=args.data_scale,
                            num_frames=5, repeats_per_sequence=1)


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 3. Setup Model (FastDVDnet)
    base_model = FastDVDnet(num_input_frames=5).to(device)
    # model = FastDVDnetR2RWrapper(base_model, alpha=args.alpha).to(device)
    model = FastDVDNetContextWrapper(base_model).to(device)

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

    
    # 5. Training Loop
    # best_val_loss = float("inf")
    # best_psnr = 0.0
    # best_epoch = -1
    # best_ckpt_path = output_dir / "best_model.pth"
    best_val_loss = float("inf")
    best_loss_epoch = -1
    best_ckpt_loss_path = output_dir / "best_model_loss.pth"

    best_psnr = 0.0
    best_psnr_epoch = -1
    best_ckpt_psnr_path = output_dir / "best_model_psnr.pth"

    train_losses, val_losses, val_psnrs = [], [], []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for x_clean_stack, x_clean in pbar:
            x_clean_stack = x_clean_stack.to(device)
            noisy_frames = [physics(x_clean_stack[:, j:j+1]) for j in range(5)]
            stack_noisy = torch.cat(noisy_frames, dim=1)  # (B, 5, H, W)
            y_central = stack_noisy[:, 2:3, :, :]

            optimizer.zero_grad()
            model.model.set_context(stack_noisy)
            x_est = model(y_central, physics, update_parameters=True)
            loss = criterion(x_est, y_central, physics, model)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        current_psnr, current_val_loss = evaluate_epoch(
        model, criterion, physics, eval_cache, device, args.eval_seed
        )
        train_losses.append(epoch_loss / len(train_dataloader))
        val_losses.append(current_val_loss)
        val_psnrs.append(current_psnr)

        print(f"Epoch {epoch+1} Val PSNR: {current_psnr:.2f} dB, "
              f"Train Loss: {train_losses[-1]:.6f}, Val Loss: {current_val_loss:.6f}")

        # if current_val_loss < best_val_loss:
        #     best_val_loss = current_val_loss
        #     best_psnr = current_psnr
        #     best_epoch = epoch + 1
        #     torch.save(model.state_dict(), best_ckpt_path)
        #     print(f"New best model saved (epoch={best_epoch}, val_loss={best_val_loss:.6f}, val_psnr={best_psnr:.2f} dB)")
        # else:
        #     print(f"No checkpoint update (best epoch={best_epoch}, val_loss={best_val_loss:.6f}, val_psnr={best_psnr:.2f} dB)")

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_loss_epoch = epoch + 1
            torch.save(model.state_dict(), best_ckpt_loss_path)
            print(f"New best Val Loss model saved (epoch={best_loss_epoch}, val_loss={best_val_loss:.6f}, val_psnr={best_psnr:.2f} dB)")

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_psnr_epoch = epoch + 1
            torch.save(model.state_dict(), best_ckpt_psnr_path)
            print(f"New best PSNR model saved (epoch={best_psnr_epoch}, val_loss={best_val_loss:.6f}, val_psnr={best_psnr:.2f} dB)")

    # with open(output_dir / "best_checkpoint.txt", "w") as f:
    #     f.write(f"best_epoch={best_epoch}\n")
    #     f.write(f"best_val_loss={best_val_loss:.6f}\n")
    #     f.write(f"best_val_psnr={best_psnr:.4f}\n")
    #     f.write(f"weights_path={best_ckpt_path}\n")
    with open(output_dir / "best_checkpoint.txt", "w") as f:
        f.write(f"best_loss_epoch={best_loss_epoch}\n")
        f.write(f"best_val_loss={best_val_loss:.6f}\n")
        f.write(f"weights_loss_path={best_ckpt_loss_path}\n")
        f.write(f"best_psnr_epoch={best_psnr_epoch}\n")
        f.write(f"best_val_psnr={best_psnr:.4f}\n")
        f.write(f"weights_psnr_path={best_ckpt_psnr_path}\n")


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
    loss_plot_path = output_dir / "loss_plot.png"
    fig.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")

    # 7. Save final result as TIF (using best-epoch weights)
    # Funciona cuando guardo solo un best_ckpt. NO BORRAR
    # if best_ckpt_path.exists():
    #     model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    #     print(f"Loaded best weights from epoch {best_epoch} for export.")
    # print("Saving final results to TIFF...")
    # model.eval()
    # with torch.no_grad():
    #     export_indices = visualize_indices if len(visualize_indices) > 0 else [0, 1, 2]
    #     for i in export_indices:
    #         if i >= len(test_dataset): break
    #         x_clean_stack, x_clean = test_dataset[i]
    #         x_clean_stack = x_clean_stack.unsqueeze(0).to(device)
    #         x_gt = x_clean.unsqueeze(0).to(device)
    #         torch.manual_seed(args.eval_seed + i)
    #         if torch.cuda.is_available():
    #             torch.cuda.manual_seed_all(args.eval_seed + i)
    #         noisy_frames = [physics(x_clean_stack[:, j:j+1]) for j in range(5)]
    #         stack_noisy = torch.cat(noisy_frames, dim=1)
    #         y_central = stack_noisy[:, 2:3, :, :]

    #         model.model.set_context(stack_noisy)
    #         x_est = model(y_central, physics)

    #         tifffile.imwrite(str(output_dir / f"fmd_{i}_denoised.tif"), x_est.squeeze().cpu().numpy().astype(np.float32))
    #         tifffile.imwrite(str(output_dir / f"fmd_{i}_noisy.tif"), y_central.squeeze().cpu().numpy().astype(np.float32))
    #         tifffile.imwrite(str(output_dir / f"fmd_{i}_clean.tif"), x_gt.squeeze().cpu().numpy().astype(np.float32))


    # Para comparar mejor PSNR y mejor Val Loss: 
    for ckpt_path, epoch, tag in [
        (best_ckpt_psnr_path, best_psnr_epoch, "psnr"),
        (best_ckpt_loss_path, best_loss_epoch, "loss"),
    ]:
        if not ckpt_path.exists():
            print(f"Checkpoint {tag} not found, skipping.")
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded best_{tag} weights from epoch {epoch} for export.")
        model.eval()
        with torch.no_grad():
            export_indices = visualize_indices if len(visualize_indices) > 0 else [0, 1, 2]
            for i in export_indices:
                if i >= len(test_dataset): break
                x_clean_stack, x_clean = test_dataset[i]
                x_clean_stack = x_clean_stack.unsqueeze(0).to(device)
                x_gt = x_clean.unsqueeze(0).to(device)
                torch.manual_seed(args.eval_seed + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.eval_seed + i)
                noisy_frames = [physics(x_clean_stack[:, j:j+1]) for j in range(5)]
                stack_noisy = torch.cat(noisy_frames, dim=1)
                y_central = stack_noisy[:, 2:3, :, :]
                model.model.set_context(stack_noisy)
                x_est = model(y_central, physics)
                tifffile.imwrite(str(output_dir / f"fmd_{i}_{tag}_denoised.tif"), x_est.squeeze().cpu().numpy().astype(np.float32))
                tifffile.imwrite(str(output_dir / f"fmd_{i}_{tag}_noisy.tif"), y_central.squeeze().cpu().numpy().astype(np.float32))
                tifffile.imwrite(str(output_dir / f"fmd_{i}_{tag}_clean.tif"), x_gt.squeeze().cpu().numpy().astype(np.float32))

    print(f"Finished. Check results in {output_dir}")

class Args:
    loss = "gr2r_mse"
    gamma = 1/255.0       # era: noise = 1/255.0
    alpha = 0.15
    epochs = 0          # era: 200
    batch_size = 16
    lr = 1e-4
    patch_size = 256      # nuevo (era variable local en train_model)
    data_scale = 1.0      # nuevo
    n_eval_sequences = None  # nuevo
    eval_seed = 42        # nuevo
    inference_dir = "results/denoising-poisson-fmdd-fastdvdnet-retry/tif_output_2026_06_09-16_44_13"# None  # setear al directorio del run anterior para saltear entrenamiento



if __name__ == "__main__":
    args = Args()
    train_model(args)
