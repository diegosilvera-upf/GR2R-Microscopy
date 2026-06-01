import torch
import os
import sys
from pathlib import Path
import numpy as np
import tifffile
from tqdm import tqdm
import deepinv as dinv

# Add current directory and Loreal directory to sys.path
sys.path.append(str(Path(".").absolute()))
sys.path.append(str(Path("../Loreal").absolute()))

from dataset import LorealDataset
from utils import linear_transform

def verify_zeroshot():
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Setup Models
    # Initialize DRUNet with same params as FMDD script
    model = dinv.models.ArtifactRemoval(
        dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained=None, nc=[16, 32, 64, 128])
    ).to(device)

    # 2. Load Checkpoint
    ckpt_path = Path("ckpts/denoising-poisson-fmd/best_model.pth")
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract state_dict if nested
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Strip "model." prefix if it exists and filter out noise_model keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        elif not k.startswith("noise_model."):
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()

    # 3. Setup Dataset
    # We'll use the path from config.sh
    loreal_data_dir = Path("/home/diegosilvera/Escritorio/2026/sequences_almost_Poisson")
    
    # Discovery of all .tif files (simple search for the first few)
    tif_files = []
    for root, dirs, files in os.walk(loreal_data_dir):
        if "check" in dirs:
            dirs.remove("check")
        for f in files:
            if f.endswith(".tif"):
                tif_files.append(os.path.join(root, f))
                if len(tif_files) >= 5: # Just take a few for the test
                    break
        if len(tif_files) >= 5:
            break

    if not tif_files:
        print(f"Error: No .tif files found in {loreal_data_dir}")
        return

    print(f"Found {len(tif_files)} .tif files for verification.")

    # 4. Results directory
    results_dir = Path("results/zeroshot")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 5. Run Inference
    data_scale = 255.0 # Match FMDD validation/training 8-bit scale
    sigma = 1/255 # FMDD training noise level
    physics = dinv.physics.Denoising(noise_model=dinv.physics.PoissonNoise(sigma))
    physics.noise_model.sigma = sigma # Ensure ArtifactRemoval finds it
    
    with torch.no_grad():
        for i, img_path in enumerate(tif_files):
            print(f"Processing image {i+1}/{len(tif_files)}: {Path(img_path).name}")
            
            # Load image
            img = tifffile.imread(img_path).astype(np.float32)
            
            # Linear transform (finding a,b from pre-processing.txt in the image's folder)
            seq_dir = Path(img_path).parent
            preproc_file = seq_dir / "pre-processing.txt"
            if preproc_file.exists():
                a, b = np.loadtxt(preproc_file)
            else:
                print(f"Warning: No pre-processing.txt found for {img_path}. Using a=1, b=0.")
                a, b = 1.0, 0.0

            # Normalize
            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device) # [1,1,H,W]
            
            use_linear_transform = False # Set to False for FMDD models which were not trained with its specific bias
            if use_linear_transform:
                img_norm = linear_transform(img_tensor, a, b, u=1) / data_scale
            else:
                img_norm = img_tensor / data_scale
                
            img_norm = torch.clamp(img_norm, min=0.0)
            print(f"  Input norm range: [{img_norm.min().item():.4f}, {img_norm.max().item():.4f}] Mean: {img_norm.mean().item():.4f}")

            # In microscopy, DRUNet often needs the input to be divisible by some power of 2
            H, W = img_norm.shape[-2:]
            new_H = (H // 16) * 16
            new_W = (W // 16) * 16
            img_norm = img_norm[:, :, :new_H, :new_W]

            # Denoise
            out = model(img_norm, physics)

            # De-normalize for saving
            out_orig_scale = out * data_scale
            # (optional: invert linear transform)
            # out_orig_scale = linear_transform(out_orig_scale, a, b, u=1, inverse=True)

            # Save
            noisy_out_path = results_dir / f"img_{i}_noisy.tif"
            denoised_out_path = results_dir / f"img_{i}_denoised.tif"
            
            tifffile.imwrite(str(noisy_out_path), img[:new_H, :new_W].astype(np.float32))
            tifffile.imwrite(str(denoised_out_path), out_orig_scale.squeeze().cpu().numpy().astype(np.float32))

    print(f"Verification finished. Results saved in {results_dir}")

if __name__ == "__main__":
    verify_zeroshot()
