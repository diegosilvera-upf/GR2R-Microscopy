r"""
Self-supervised learning with Generalized Recorrupted-to-Recovered (GR2R)
Training Script for Loreal Dataset (Skin/Biological Sequences) using DRUNet
====================================================================================================
"""

import os
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

from loreal_dataset import get_valid_sequences, LorealSequenceDataset
import matplotlib.pyplot as plt


# # Add Loreal directory to sys.path to import dataset utilities
# sys.path.append(str(Path("../Loreal").absolute()))
# sys.path.append(str(Path("/home/diegosilvera/Escritorio/2026").resolve()))

# from dataset import LorealDataset, get_valid_sequences, FastDVDnetDataset
# # Import linear_transform but we'll use it carefully or skip if it causes bias
# from utils import linear_transform
from models_FastDVDnet_sans_noise_map import FastDVDnet

##### Esta clase la comento, quedo complicada de más y capaz rompe cosas
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
#         # Following exactly the logic in deepinv.loss.r2r.py:
#         # y1 = gamma * (z - Binomial(z, alpha)) / (1 - alpha)
#         if self.training:
#             with torch.no_grad():
#                 gain = physics.noise_model.gain if (physics is not None and hasattr(physics.noise_model, 'gain')) else args.noise
#                 for i in [0, 1, 3, 4]:
#                     y_neighbor = stack[:, i:i+1, :, :]
#                     z = y_neighbor / gain
#                     # Note: alpha is the probability of removal in deepinv's set_binomial_corruptor
#                     sampler = torch.distributions.Binomial(torch.clamp(torch.round(z), min=0), self.alpha)
#                     stack[:, i:i+1, :, :] = gain * (z - sampler.sample()) / (1.0 - self.alpha)
        
#         # Replace central frame with the (already recorrupted) y_central
#         stack[:, 2:3, :, :] = y_central
        
#         # FastDVDnet handles the forward pass
#         return self.model(stack)

# Me la hizo Claude, es mucho más directa.
# Deepinv se encarga de agregar ruido, acá solo aislo el frame central
class FastDVDNetContextWrapper(torch.nn.Module):
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
PROJECT_NAME = "denoising-poisson-loreal-fastdvdnet-retry"
# Directory where image sequences are stored
LOREAL_DATA_DIR = Path("/home/diegosilvera/Escritorio/2026/sequences_almost_Poisson")
RESULTS_DIR = BASE_DIR / "results" / PROJECT_NAME
CKPT_DIR = BASE_DIR / "ckpts" / PROJECT_NAME
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_FILE = BASE_DIR / "txts/loreal_split.txt"

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

# Idem que en DRUnet + Loreal
def load_loreal_split(valid_sequences, split_file, val_prefixes=None, test_prefixes=None):
    """
    Returns train/val/test split using:
      1) explicit split file when available
      2) otherwise, prefix-based fallback for val/test
    """
    seq_by_name = {Path(seq_path).name: (seq_path, a, b) for seq_path, a, b in valid_sequences}

    train_seq = []
    val_seq = []
    test_seq = []
    visualize_names = []

    if split_file.exists():
        print(f"Loading split from {split_file}")
        with open(split_file, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if (not line) or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                seq_name = parts[0]
                role = parts[1].lower()
                visualize = len(parts) >= 3 and parts[2].lower() == "true"

                if seq_name not in seq_by_name:
                    print(f"  WARNING: {seq_name} in split file was not found in valid Loreal sequences.")
                    continue

                item = seq_by_name[seq_name]
                if role == "val":
                    val_seq.append(item)
                elif role == "test":
                    test_seq.append(item)
                else:
                    train_seq.append(item)

                if visualize:
                    visualize_names.append(seq_name)

        assigned = {Path(s[0]).name for s in train_seq + val_seq + test_seq}
        leftovers = [item for item in valid_sequences if Path(item[0]).name not in assigned]
        train_seq.extend(leftovers)
        if leftovers:
            print(f"Added {len(leftovers)} unassigned sequences to train.")
    else:
        print("No split file found. Using prefix fallback split.")
        val_prefixes = val_prefixes or []
        test_prefixes = test_prefixes or []

        def starts_with_any(name, prefixes):
            return any(name.startswith(prefix) for prefix in prefixes)

        for item in valid_sequences:
            name = Path(item[0]).name
            if starts_with_any(name, test_prefixes):
                test_seq.append(item)
            elif starts_with_any(name, val_prefixes):
                val_seq.append(item)
            else:
                train_seq.append(item)

        if len(val_seq) == 0:
            n_train = int(0.9 * len(valid_sequences))
            train_seq = valid_sequences[:n_train]
            val_seq = valid_sequences[n_train:]
            test_seq = []
            print("Fallback had no val matches, using deterministic 90/10 train/val split.")

    return train_seq, val_seq, test_seq, visualize_names

#Idem que DRUnet + Loreal, pero agarrando el frame central
def evaluate_val_loss(model, val_loader, criterion, physics, eval_seed):
    model.eval()
    running = 0.0
    n_batches = 0
    with torch.no_grad():
        for y_stack, _ in val_loader:
            y_stack = y_stack.to(device)
            y_central = y_stack[:, 2:3, :, :] #Diferencia con la de DRUnet
            torch.manual_seed(eval_seed + n_batches)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_seed + n_batches)
            model.model.set_context(y_stack)
            model.training = True
            x_est = model(y_central, physics, update_parameters=True)
            loss_val = criterion(x_est, y_central, physics, model)
            model.training = False
            running += loss_val.item()
            n_batches += 1
    if n_batches == 0:
        return float("inf")
    return running / n_batches

#Idem que DRUnet + Loreal, pero usando secuencias y haciendo denoising solo del frame central
def export_sequences(model, physics, sequences, output_dir, tag, max_sequences=2, data_scale=255.0):
    if len(sequences) == 0:
        print(f"No sequences to export for tag={tag}.")
        return
    chosen = sequences[:max_sequences]
    print(f"Exporting {len(chosen)} {tag} sequence(s) as TIFF stacks.")
    model.eval()
    with torch.no_grad():
        for seq_path, a, b in chosen:
            seq_name = Path(seq_path).name
            seq_ds = LorealSequenceDataset(
                sequence_info=[(seq_path, a, b)],
                num_frames=5,
                data_scale=data_scale,
            )
            if len(seq_ds) == 0:
                print(f"  WARNING: No valid stacks for {seq_name}, skipping.")
                continue
            denoised_frames = []
            noisy_frames = []
            for i in tqdm(range(len(seq_ds)), desc=f"{tag}:{seq_name}"):
                y_stack, y_central = seq_ds[i]
                y_stack = y_stack.unsqueeze(0).to(device)
                y_central = y_central.unsqueeze(0).to(device)
                model.model.set_context(y_stack)
                x_est = model(y_central, physics)
                denoised_frames.append(x_est.squeeze().cpu().numpy().astype(np.float32))
                noisy_frames.append(y_central.squeeze().cpu().numpy().astype(np.float32))
            denoised_stack = np.stack(denoised_frames, axis=0)
            noisy_stack = np.stack(noisy_frames, axis=0)
            safe_name = seq_name.replace("/", "_")
            tifffile.imwrite(str(output_dir / f"{tag}_{safe_name}_denoised.tif"), denoised_stack)
            tifffile.imwrite(str(output_dir / f"{tag}_{safe_name}_noisy.tif"), noisy_stack)
            print(f"Saved {tag}_{safe_name}_denoised.tif with shape {denoised_stack.shape}")


def train_model(args):

    if args.inference_dir is not None:
        output_dir = Path(args.inference_dir)
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        output_dir = RESULTS_DIR / f"tif_output_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_parameters(args, output_dir)

    print(f"Starting training on {device} with {args.loss} loss...")

    # 1. Physics and noise settings
    noise_model = dinv.physics.PoissonNoise(args.gamma)
    noise_model.sigma = args.gamma
    physics = dinv.physics.Denoising(noise_model=noise_model)

    # 2. Sequence discovery and split
    sequence_paths = sorted(LOREAL_DATA_DIR.glob("*"))
    valid_sequences = get_valid_sequences(sequence_paths)
    print(f"Found {len(valid_sequences)} valid Loreal sequences.")

    train_seq, val_seq, test_seq, visualize_names = load_loreal_split(
        valid_sequences=valid_sequences,
        split_file=SPLIT_FILE,
        val_prefixes=args.val_prefixes,
        test_prefixes=args.test_prefixes,
    )
    print(f"Split: {len(train_seq)} train / {len(val_seq)} val / {len(test_seq)} test")

    # 3. Datasets — data is already noisy (real Loreal sequences), no physics needed for noise
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    train_dataset = LorealSequenceDataset(
        sequence_info=train_seq,
        patch_size=(args.patch_size, args.patch_size),
        transform=transform,
        num_frames=5,
        data_scale=args.data_scale,
        repeats_per_frame=args.repeats_per_frame,
    )
    val_dataset = LorealSequenceDataset(
        sequence_info=val_seq,
        patch_size=(args.patch_size, args.patch_size),
        transform=None,
        num_frames=5,
        data_scale=args.data_scale,
        repeats_per_frame=1,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers
    )
    n_train_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    print(
        f"Train items: {len(train_dataset)} (~{n_train_batches} batches/epoch, "
        f"repeats_per_frame={args.repeats_per_frame})"
    )
    print(f"Val items: {len(val_dataset)}")

    # 4. Model
    base_model = FastDVDnet(num_input_frames=5).to(device)
    model = FastDVDNetContextWrapper(base_model).to(device)

    if args.pretrained_ckpt:
        ckpt_path = Path(args.pretrained_ckpt)
        if ckpt_path.exists():
            print(f"Loading pre-trained weights from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            cleaned = {k: v for k, v in state_dict.items()
                       if not k.startswith("noise_model.")}
            model.load_state_dict(cleaned, strict=False)
        else:
            print(f"WARNING: pretrained_ckpt not found: {ckpt_path}")

    if args.loss != "gr2r_mse":
        raise ValueError("Only gr2r_mse is supported in this script version")

    criterion = R2RLoss(noise_model=noise_model, alpha=args.alpha)
    model = criterion.adapt_model(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 5. Training loop — val_loss on real noisy data (no GT available for PSNR)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    best_epoch = -1
    best_ckpt_path = output_dir / "best_model.pth"

    for epoch in range(args.epochs):
        model.train()
        running_train = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for y_stack, _ in pbar:
            y_stack = y_stack.to(device)
            y_central = y_stack[:, 2:3, :, :]

            optimizer.zero_grad()
            model.model.set_context(y_stack)
            x_est = model(y_central, physics, update_parameters=True)
            loss = criterion(x_est, y_central, physics, model)
            loss.backward()
            optimizer.step()

            running_train += loss.item()
            pbar.set_postfix({"train_loss": f"{loss.item():.5f}"})

        train_loss_epoch = running_train / max(len(train_loader), 1)
        val_loss_epoch = evaluate_val_loss(model, val_loader, criterion, physics, args.eval_seed)

        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss_epoch)
        print(
            f"Epoch {epoch+1}: train_loss={train_loss_epoch:.6f}, "
            f"val_loss={val_loss_epoch:.6f}"
        )

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_ckpt_path)
            print(
                f"New best model saved to {best_ckpt_path} "
                f"(epoch={best_epoch}, val_loss={best_val_loss:.6f})"
            )

        if (epoch + 1) % args.checkpoint_every == 0:
            periodic_path = output_dir / f"model_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), periodic_path)
            print(f"Periodic checkpoint saved: {periodic_path}")

    # 6. Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loreal FastDVDNet training/validation loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = output_dir / "loss_plot.png"
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")

    with open(output_dir / "best_checkpoint.txt", "w") as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_loss={best_val_loss}\n")
        f.write(f"weights_path={best_ckpt_path}\n")
    print(f"Best checkpoint metadata saved to {output_dir / 'best_checkpoint.txt'}")

    # 7. Export selected sequences
    explicit_viz = [s for s in val_seq + test_seq if Path(s[0]).name in set(visualize_names)]
    if len(explicit_viz) > 0:
        export_sequences(model, physics, explicit_viz, output_dir, tag="viz",
                         max_sequences=args.max_export_sequences, data_scale=args.data_scale)
    else:
        export_sequences(model, physics, val_seq, output_dir, tag="val",
                         max_sequences=args.max_export_sequences, data_scale=args.data_scale)
        export_sequences(model, physics, test_seq, output_dir, tag="test",
                         max_sequences=args.max_export_sequences, data_scale=args.data_scale)

    print(f"Finished. Check results in {output_dir}")


class Args:
    loss = "gr2r_mse"
    gamma = 1/255.0
    alpha = 0.15
    epochs = 100
    batch_size = 16
    val_batch_size = 16
    lr = 1e-4
    patch_size = 256
    data_scale = 255.0
    num_workers = 4
    repeats_per_frame = 10
    eval_seed = 43
    checkpoint_every = 10
    pretrained_ckpt = 'FastDVDnet-pure_poisson-a=1-normalization_by_255.pth'# None  # path al checkpoint FMDD FastDVDNet para transfer learning
    val_prefixes = ["HF1_", "Mela1_"]
    test_prefixes = ["HF2_", "Mela2_"]
    max_export_sequences = 2
    inference_dir = None


if __name__ == "__main__":
    args = Args()
    train_model(args)

##################################################################################
        # 6. Evaluation (Visual check) — DEPRECATED
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
