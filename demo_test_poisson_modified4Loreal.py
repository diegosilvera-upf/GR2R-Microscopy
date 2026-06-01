r"""
Self-supervised learning with Generalized Recorrupted-to-Recovered (GR2R)
Training Script for Loreal Dataset (Microscopy Denoising) using DRUNet
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

# # Add Loreal directory to sys.path to import dataset utilities
# sys.path.append(str(Path("../Loreal").absolute()))
# sys.path.append(str(Path("/home/diegosilvera/Escritorio/2026").resolve()))
# from dataset import LorealDataset, get_valid_sequences
# # Import linear_transform but we'll use it carefully or skip if it causes bias
# from utils import linear_transform
from loreal_dataset import get_valid_sequences, LorealSequenceDataset


# ---------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------
BASE_DIR = Path(".")
PROJECT_NAME = "denoising-poisson-loreal-drunet"
DATA_DIR = Path("/home/diegosilvera/Escritorio/2026/sequences_almost_Poisson")
RESULTS_DIR = BASE_DIR / "results" / PROJECT_NAME
CKPT_DIR = BASE_DIR / "ckpts" / PROJECT_NAME
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

def save_parameters(args, output_dir):
    """Save experiment parameters to a text file in the output directory."""
    with open(output_dir / "parameters.txt", "w") as f:
        f.write(f"Experiment: demo_test_poisson_modified4Loreal.py\n")
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

# def _get_tif_files_from_sequences(sequences):
#     """Extract individual .tif file paths from a list of (seq_path, a, b) tuples."""
#     tif_files = []
#     for seq_path, a, b in sequences:
#         seq = Path(seq_path)
#         tif_files.extend(sorted(seq.glob("*.tif")))
#     return [str(f) for f in sorted(tif_files)]

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
    # Split by SEQUENCE (same logic as FastDVDNet script for comparable results)
    # seq_dirs = [d for d in LOREAL_DATA_DIR.iterdir() if d.is_dir() and d.name != "check"]
    # valid_sequences = get_valid_sequences(seq_dirs)
    sequence_paths = sorted(DATA_DIR.glob("*"))
    valid_sequences = get_valid_sequences(sequence_paths)
    print(f"Found {len(valid_sequences)} valid sequences.")

    # Same 90/10 sequence-based split as FastDVDNet script
    n_total = len(valid_sequences)
    n_train = int(0.9 * n_total)
    train_seq = valid_sequences[:n_train]
    test_seq = valid_sequences[n_train:]
    print(f"Train sequences: {n_train}, Test sequences: {n_total - n_train}")

    # Data Augmentation (D4 flips)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    patch_size = 128
    data_scale = 255.0
    
    # Use LorealSequenceDataset with num_frames=1 for 2D DRUNet
    train_dataset = LorealSequenceDataset(train_seq, patch_size=(patch_size, patch_size), 
                                         transform=transform, num_frames=1, data_scale=data_scale)

    # # Simple wrapper to handle normalization and patching
    # class LorealWrapper(LorealDataset):
    #     def __getitem__(self, idx):
    #         img = super().__getitem__(idx) # Returns [H,W] tensor
    #         if img.ndim == 2:
    #             img = img.unsqueeze(0) # [1, H, W]
            
    #         # Normalization (Matches successful zero-shot verification)
    #         img = img / 255.0 
    #         img = torch.clamp(img, min=0.0, max=1.0)

    #         # We need to return (y, y) or similar if we don't have GT. 
    #         # R2RLoss expects (x, y) where x is clean, but during self-supervised
    #         # training it handles y as input.
    #         return img, img

    # # train_dataset = LorealWrapper(image_paths=train_files, patch_size=(patch_size, patch_size))
    # # test_dataset = LorealWrapper(image_paths=test_files) # No patching for test to see full images (or can pad)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Train items: {len(train_dataset)}")
    
    # 3. Setup Model (1 channel, same as FMDD successful run)
    model = dinv.models.ArtifactRemoval(
        dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained=None, nc=[16, 32, 64, 128])
    ).to(device)

    # Optional: Load pre-trained FMDD weights as starting point
    if args.pretrained_ckpt:
        print(f"Loading pre-trained weights from {args.pretrained_ckpt}...")
        checkpoint = torch.load(args.pretrained_ckpt, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."): new_state_dict[k[6:]] = v
            elif not k.startswith("noise_model."): new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    # 3. Setup Model (1 channel)
    model = dinv.models.ArtifactRemoval(dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained=None, nc=[16, 32, 64, 128])).to(device)

    # 4. Setup Loss and Optimizer
    if args.loss == "gr2r_mse":
        criterion = R2RLoss(noise_model=noise_model, alpha=args.alpha)
        model = criterion.adapt_model(model)
    else:
        raise ValueError("Only gr2r_mse is supported in this script version")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. Training Loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for y_stack, y in pbar:
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

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pth")
            print(f"New best model saved! (Loss: {best_loss:.6f})")

    # --- Fixed evaluation sequences (same as FastDVDNet script) ---
    EVAL_SEQ_PREFIXES = ["HF1_", "Mela1_"]
    print("\n--- Evaluation on fixed sequences ---")
    print(f"Test sequences: {[Path(s[0]).name for s in test_seq]}")
    
    model.eval()
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
            
            # Iterate over all frames in the sequence
            tif_files = sorted(Path(seq_path).glob("*.tif"))
            print(f"    Processing {len(tif_files)} frames...")
            
            denoised_frames = []
            orig_frames = []
            
            for i, eval_file in enumerate(tqdm(tif_files, desc=f"Denoising {tag}")):
                # Load and preprocess
                img = tifffile.imread(str(eval_file)).astype(np.float32)
                y_test = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
                y_test = y_test / 255.0
                y_test = torch.clamp(y_test, min=0.0, max=1.0)
                
                # Crop to multiple of 16 for DRUNet
                H, W = y_test.shape[-2:]
                new_H, new_W = (H // 16) * 16, (W // 16) * 16
                y_test = y_test[:, :, :new_H, :new_W]
                
                x_est = model(y_test, physics)
                
                denoised_frames.append(x_est.squeeze().cpu().detach().numpy().astype(np.float32))
                orig_frames.append(y_test.squeeze().cpu().detach().numpy().astype(np.float32))
            
            # Save stacks
            denoised_stack = np.stack(denoised_frames, axis=0)
            orig_stack = np.stack(orig_frames, axis=0)
            
            tifffile.imwrite(str(output_dir / f"loreal_{tag}_denoised.tif"), denoised_stack)
            tifffile.imwrite(str(output_dir / f"loreal_{tag}_orig.tif"), orig_stack)
            print(f"    Saved: loreal_{tag}_denoised.tif ({denoised_stack.shape})")
    
    print(f"Finished. Check results in {output_dir}")

class Args:
    loss = "gr2r_mse"
    noise = 1/255 # Gain used in FMDD training
    epochs = 3
    batch_size = 32 # Reduced batch size for 1024x1024 or large patches
    lr = 1e-4
    alpha = 0.15 # Parameter for R2R thinning
    pretrained_ckpt = "ckpts/denoising-poisson-fmd/best_model.pth"

if __name__ == "__main__":
    args = Args()
    train_model(args)


##################################################################################

        # 6. Evaluation (Visual check)
        # model.eval()
        # print(f"Epoch {epoch+1} Evaluation...")
        # with torch.no_grad():
        #     # Process one test image
        #     y_test, _ = test_dataset[0] #Lo estoy cambiando para ir probando en distintas secuencias
        #     y_test = y_test.unsqueeze(0).to(device)
            
        #     # Crop to multiple of 16 for DRUNet
        #     H, W = y_test.shape[-2:]
        #     new_H, new_W = (H // 16) * 16, (W // 16) * 16
        #     y_test = y_test[:, :, :new_H, :new_W]
            
        #     x_est = model(y_test, physics)
            
        #     # Save output
        #     tifffile.imwrite(str(output_dir / f"epoch_{epoch}_denoised.tif"), (x_est * 255).squeeze().cpu().numpy().astype(np.float32))
        #     if epoch == 0:
        #         tifffile.imwrite(str(output_dir / f"input_noisy_{epoch}.tif"), (y_test * 255).squeeze().cpu().numpy().astype(np.float32))

##################################################################################

                    # with torch.no_grad():
        #     # Process one test image
        #     y_test, _ = test_dataset[0] #Lo estoy cambiando para ir probando en distintas secuencias
        #     y_test = y_test.unsqueeze(0).to(device)
            
        #     # Crop to multiple of 16 for DRUNet
        #     H, W = y_test.shape[-2:]
        #     new_H, new_W = (H // 16) * 16, (W // 16) * 16
        #     y_test = y_test[:, :, :new_H, :new_W]
            
        #     x_est = model(y_test, physics)
            
        #     # Save output
        #     tifffile.imwrite(str(output_dir / f"epoch_{epoch}_denoised.tif"), (x_est * 255).squeeze().cpu().numpy().astype(np.float32))
        #     if epoch == 0:
        #         tifffile.imwrite(str(

            # ###Pisada para correr el modelo en 023, no aporta otra cosa este bloque ###
    # y_test, _ = train_dataset[3] #Es para usar alguna imagen de las carpetas numeradas, eliminar después
    # y_test = y_test.unsqueeze(0).to(device)
    
    # # Crop to multiple of 16 for DRUNet
    # H, W = y_test.shape[-2:]
    # new_H, new_W = (H // 16) * 16, (W // 16) * 16
    # y_test = y_test[:, :, :new_H, :new_W]
    
    # x_est = model(y_test, physics)
    
    # tifffile.imwrite(str(output_dir / f"loreal_train3_denoised.tif"), x_est.squeeze().cpu().detach().numpy().astype(np.float32))
    # # tifffile.imwrite(str(output_dir / f"fmd_{i}_noisy.tif"), y_noisy.squeeze().cpu().detach().numpy().astype(np.float32))
    # tifffile.imwrite(str(output_dir / f"loreal_train3_orig.tif"), y_test.squeeze().cpu().detach().numpy().astype(np.float32))
    # ###Fin de la pisada###