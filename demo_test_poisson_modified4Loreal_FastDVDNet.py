r"""
Self-supervised learning with Generalized Recorrupted-to-Recovered (GR2R)
Training Script for Loreal Dataset (Skin/Biological Sequences) using DRUNet
====================================================================================================
"""

import os
import sys
from datetime import datetime
import deepinv as dinv
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from torchvision import transforms
from deepinv.loss import R2RLoss
import tifffile
from tqdm import tqdm

# Add Loreal directory to sys.path to import dataset utilities
sys.path.append(str(Path("../Loreal").absolute()))
sys.path.append(str(Path("/home/diegosilvera/Escritorio/2026").resolve()))

from dataset import LorealDataset, get_valid_sequences, FastDVDnetDataset
# Import linear_transform but we'll use it carefully or skip if it causes bias
from utils import linear_transform
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
        # Following exactly the logic in deepinv.loss.r2r.py:
        # y1 = gamma * (z - Binomial(z, alpha)) / (1 - alpha)
        if self.training:
            with torch.no_grad():
                gain = physics.noise_model.gain if (physics is not None and hasattr(physics.noise_model, 'gain')) else args.noise
                for i in [0, 1, 3, 4]:
                    y_neighbor = stack[:, i:i+1, :, :]
                    z = y_neighbor / gain
                    # Note: alpha is the probability of removal in deepinv's set_binomial_corruptor
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
PROJECT_NAME = "denoising-poisson-loreal-fastdvdnet"
# Directory where image sequences are stored
LOREAL_DATA_DIR = Path("/home/diegosilvera/Escritorio/2026/sequences_almost_Poisson")
RESULTS_DIR = BASE_DIR / "results" / PROJECT_NAME
CKPT_DIR = BASE_DIR / "ckpts" / PROJECT_NAME
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

def save_parameters(args, output_dir):
    """Save experiment parameters to a text file in the output directory."""
    with open(output_dir / "parameters.txt", "w") as f:
        f.write(f"Experiment: demo_test_poisson_modified4Loreal_FastDVDNet.py\n")
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
    
    # Save command to log file
    with open(RESULTS_DIR / "command.txt", "w") as f:
        f.write(" ".join(sys.argv))
    
    # 1. Setup Noise and Physics
    noise_model = dinv.physics.PoissonNoise(args.noise)
    noise_model.sigma = args.noise # Crucial for some models
    physics = dinv.physics.Denoising(noise_model=noise_model)

    # 2. Setup Datasets
    # Discovery of sequences
    seq_dirs = [d for d in LOREAL_DATA_DIR.iterdir() if d.is_dir() and d.name != "check"]
    valid_sequences = get_valid_sequences(seq_dirs)
    print(f"Found {len(valid_sequences)} valid sequences.")

    # Split into train/test
    n_total = len(valid_sequences)
    n_train = int(0.9 * n_total)
    train_seq = valid_sequences[:n_train]
    test_seq = valid_sequences[n_train:]

    # Use actual sequence dataset for FastDVDnet
    patch_size = 256
    # data_scale=255.0 to match the DRUNet run that worked (avoids extremely small values)
    train_dataset = FastDVDnetDataset(sequence_info=train_seq, patch_size=(patch_size, patch_size), data_scale=255.0)
    test_dataset = FastDVDnetDataset(sequence_info=test_seq, data_scale=255.0)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # 3. Setup Model (FastDVDnet)
    base_model = FastDVDnet(num_input_frames=5).to(device)
    model = FastDVDnetR2RWrapper(base_model, alpha=args.alpha).to(device)

    # Optional: Load pre-trained weights as starting point
    if args.pretrained_ckpt:
        print(f"Loading pre-trained weights from {args.pretrained_ckpt}...")
        checkpoint = torch.load(args.pretrained_ckpt, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."): new_state_dict[k[6:]] = v
            elif not k.startswith("noise_model."): new_state_dict[k] = v
        
        # Load weights into the inner FastDVDnet model, not the wrapper
        base_model.load_state_dict(new_state_dict, strict=False)
        print("Pre-trained weights loaded successfully into base_model.")
        
    # --- Initial zero-shot evaluation ---
    print("Initial Zero-shot Evaluation...")
    model.eval()
    with torch.no_grad():
        stack_test, y_test = test_dataset[3]
        stack_test = stack_test.unsqueeze(0).to(device)
        y_test = y_test.unsqueeze(0).to(device)
        if hasattr(model, "set_context"):
            model.set_context(stack_test)
        x_est = model(y_test, physics)
        tifffile.imwrite(str(output_dir / "zero_shot_denoised.tif"), (x_est * 255).squeeze().cpu().numpy().astype(np.float32))
        tifffile.imwrite(str(output_dir / "input_noisy_initial.tif"), (y_test * 255).squeeze().cpu().numpy().astype(np.float32))
    # ------------------------------------

    # 4. Setup Loss and Optimizer
    if args.loss == "gr2r_mse":
        criterion = R2RLoss(noise_model=noise_model, alpha=args.alpha)
        model = criterion.adapt_model(model)
    else:
        raise ValueError("Only gr2r_mse is supported in this script version")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    
    # 5. Training Loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for stack, y in pbar:
            stack, y = stack.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Set context for video R2R. If model is adapted, it's wrapped in R2RModel.
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
        scheduler.step() 

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pth")
            print(f"New best model saved! (Loss: {best_loss:.6f})")

    # --- Fixed evaluation sequences (same as DRUNet script) ---
    EVAL_SEQ_PREFIXES = ["HF1_", "Mela1_"]
    print("\n--- Evaluation on fixed sequences ---")
    print(f"Test sequences: {[Path(s[0]).name for s in test_seq]}")
    
    # Use inner model for clean inference (no R2R corruption)
    inner_model = model.model if hasattr(model, "model") else model
    inner_model.eval()
    with torch.no_grad():
        for prefix in EVAL_SEQ_PREFIXES:
            # Find the sequence in ALL valid sequences (it may be in train or test)
            match = [(p, a, b) for p, a, b in valid_sequences if Path(p).name.startswith(prefix)]
            if not match:
                print(f"  WARNING: No sequence found with prefix '{prefix}', skipping.")
                continue
            seq_path, a, b = match[0]
            seq_name = Path(seq_path).name
            tag = prefix.rstrip("_")
            print(f"  Evaluating: {seq_name}")
            
            # Build a temporary FastDVDnetDataset for this sequence only
            eval_ds = FastDVDnetDataset(sequence_info=[match[0]], data_scale=255.0)
            if len(eval_ds) == 0:
                print(f"    WARNING: No valid stacks in {seq_name}, skipping.")
                continue
            
            print(f"    Processing {len(eval_ds)} stacks...")
            denoised_frames = []
            orig_frames = []
            
            for i in tqdm(range(len(eval_ds)), desc=f"Denoising {tag}"):
                stack_test, y_test = eval_ds[i]
                stack_test = stack_test.unsqueeze(0).to(device)
                y_test = y_test.unsqueeze(0).to(device)
                
                # Crop to multiple of 16
                H, W = y_test.shape[-2:]
                new_H, new_W = (H // 16) * 16, (W // 16) * 16
                y_test = y_test[:, :, :new_H, :new_W]
                stack_test = stack_test[:, :, :new_H, :new_W]
                
                inner_model.set_context(stack_test)
                x_est = inner_model(y_test, physics)
                
                denoised_frames.append(x_est.squeeze().cpu().detach().numpy().astype(np.float32))
                orig_frames.append(y_test.squeeze().cpu().detach().numpy().astype(np.float32))
            
            # Save stacks
            denoised_stack = np.stack(denoised_frames, axis=0)
            orig_stack = np.stack(orig_frames, axis=0)
            
            tifffile.imwrite(str(output_dir / f"loreal_{tag}_denoised.tif"), denoised_stack)
            tifffile.imwrite(str(output_dir / f"loreal_{tag}_orig.tif"), orig_stack)
            print(f"    Saved: loreal_{tag}_denoised.tif ({denoised_stack.shape})")
    

    print(f"Finished. Check results in {RESULTS_DIR}")

class Args:
    loss = "gr2r_mse"
    #noise = 1/255 # Gain used in FMDD training
    epochs = 3
    batch_size = 16 # Reduced batch size for 1024x1024 or large patches
    lr = 5e-5
    alpha = 0.1 # Parameter for R2R thinning
    # noise = 1.0 because after linear_transform(u=1), images are in "counts" space
    # but then we divide by 255.0, so the new gain is 1/255.0
    noise = 1/255.0 
    pretrained_ckpt = "/home/diegosilvera/Escritorio/2026/FastDVDnet-pure_poisson-a=1-normalization_by_255.pth" #None # DRUNet weights are not compatible with FastDVDnet

if __name__ == "__main__":
    args = Args()
    train_model(args)

##################################################################################
        # 6. Evaluation (Visual check)
        # model.eval()
        # print(f"Epoch {epoch+1} Evaluation...")
        # with torch.no_grad():
        #     # Process one test stack
        #     stack_test, y_test = test_dataset[3] # Evaluation on a fixed sequence
        #     stack_test = stack_test.unsqueeze(0).to(device)
        #     y_test = y_test.unsqueeze(0).to(device)
            
        #     # Use inner model if adapted
        #     inner_model = model.model if hasattr(model, "model") else model
        #     inner_model.set_context(stack_test)
        #     x_est = inner_model(y_test, physics)
            
        #     # Save output
        #     tifffile.imwrite(str(output_dir / f"epoch_{epoch}_denoised.tif"), (x_est * 255).squeeze().cpu().numpy().astype(np.float32))
        #     if epoch < 10:
        #         tifffile.imwrite(str(output_dir / f"input_noisy_{epoch}.tif"), (y_test * 255).squeeze().cpu().numpy().astype(np.float32))
