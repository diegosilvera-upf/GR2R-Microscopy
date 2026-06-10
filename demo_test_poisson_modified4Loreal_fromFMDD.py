r"""
Self-supervised learning with Generalized Recorrupted-to-Recovered (GR2R)
Training Script for Loreal real sequences using DRUNet
====================================================================================================
"""

from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import deepinv as dinv
from deepinv.loss import R2RLoss

from loreal_dataset import get_valid_sequences, LorealSequenceDataset


# ---------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------
BASE_DIR = Path(".")
PROJECT_NAME = "denoising-poisson-loreal-drunet-from-fmdd"
DATA_DIR = Path("/home/diegosilvera/Escritorio/2026/sequences_almost_Poisson")
RESULTS_DIR = BASE_DIR / "results" / PROJECT_NAME
CKPT_DIR = BASE_DIR / "ckpts" / PROJECT_NAME
SPLIT_FILE = BASE_DIR / "loreal_split.txt"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


def save_parameters(args, output_dir):
    """Save experiment parameters to a text file in the output directory."""
    with open(output_dir / "parameters.txt", "w") as f:
        f.write("Experiment: demo_test_poisson_modified4Loreal_fromFMDD.py\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write("-" * 50 + "\n")
        for key in dir(args):
            if key.startswith("_"):
                continue
            value = getattr(args, key)
            if not callable(value):
                f.write(f"{key} = {value}\n")
    print(f"Parameters saved to {output_dir / 'parameters.txt'}")


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


def _to_model_input(y_stack):
    """Map Loreal dataset output [B, num_frames, H, W] to DRUNet input [B, 1, H, W]."""
    return y_stack[:, 0:1, :, :] #Devuelve el primer frame de la secuencia
                                 #Para el caso de num_frames=1, es lo mismo que devolver y_stack
                                 #Pero para el caso de num_frames=5, debería cambiarlo a y_stack[:, 2:3, :, :]


def evaluate_val_loss(model, val_loader, criterion, physics, eval_seed):
    model.eval()
    running = 0.0
    n_batches = 0

    with torch.no_grad():
        for y_stack, _ in val_loader:
            y_stack = y_stack.to(device)
            y = _to_model_input(y_stack)

            torch.manual_seed(eval_seed + n_batches)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_seed + n_batches)

            model.training = True
            x_est = model(y, physics, update_parameters=True)
            loss_val = criterion(x_est, y, physics, model)
            model.training = False

            running += loss_val.item()
            n_batches += 1

    if n_batches == 0:
        return float("inf")
    return running / n_batches


def export_sequences(model, physics, sequences, output_dir, tag, max_sequences=2):
    if len(sequences) == 0:
        print(f"No sequences to export for tag={tag}.")
        return

    chosen = sequences[:max_sequences]
    print(f"Exporting {len(chosen)} {tag} sequence(s) as full TIFF stacks.")

    model.eval()
    with torch.no_grad():
        for seq_path, _, _ in chosen:
            seq_name = Path(seq_path).name
            tif_files = sorted(Path(seq_path).glob("*.tif"))
            if len(tif_files) == 0:
                continue

            denoised_frames = []
            noisy_frames = []
            for frame_path in tqdm(tif_files, desc=f"{tag}:{seq_name}"):
                img = tifffile.imread(str(frame_path)).astype(np.float32)
                y = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device) / 255.0
                y = torch.clamp(y, min=0.0, max=1.0)

                h, w = y.shape[-2:]
                h16 = (h // 16) * 16
                w16 = (w // 16) * 16
                y = y[:, :, :h16, :w16]

                x_est = model(y, physics)
                denoised_frames.append(x_est.squeeze().cpu().numpy().astype(np.float32))
                noisy_frames.append(y.squeeze().cpu().numpy().astype(np.float32))

            denoised_stack = np.stack(denoised_frames, axis=0)
            noisy_stack = np.stack(noisy_frames, axis=0)
            safe_name = seq_name.replace("/", "_")

            tifffile.imwrite(str(output_dir / f"{tag}_{safe_name}_denoised.tif"), denoised_stack)
            tifffile.imwrite(str(output_dir / f"{tag}_{safe_name}_noisy.tif"), noisy_stack)
            print(f"Saved {tag}_{safe_name}_denoised.tif with shape {denoised_stack.shape}")


def train_model(args):
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = RESULTS_DIR / f"tif_output_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_parameters(args, output_dir)

    print(f"Starting training on {device} with {args.loss} loss...")

    # 1) Physics and noise settings
    noise_model = dinv.physics.PoissonNoise(args.noise)
    noise_model.sigma = args.noise
    physics = dinv.physics.Denoising(noise_model=noise_model) #Defino el operador de degradación, pero no lo uso para agregar ruido

    # 2) Sequence discovery and split
    sequence_paths = sorted(DATA_DIR.glob("*"))
    valid_sequences = get_valid_sequences(sequence_paths)
    print(f"Found {len(valid_sequences)} valid Loreal sequences.")

    train_seq, val_seq, test_seq, visualize_names = load_loreal_split(
        valid_sequences=valid_sequences,
        split_file=SPLIT_FILE,
        val_prefixes=args.val_prefixes,
        test_prefixes=args.test_prefixes,
    )
    print(f"Split: {len(train_seq)} train / {len(val_seq)} val / {len(test_seq)} test")

    # 3) Datasets
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    train_dataset = LorealSequenceDataset(
        sequence_info=train_seq,
        patch_size=(args.patch_size, args.patch_size),
        transform=transform,
        num_frames=1,
        data_scale=args.data_scale,
        repeats_per_frame=args.repeats_per_frame,
    )
    val_dataset = LorealSequenceDataset(
        sequence_info=val_seq,
        patch_size=(args.patch_size, args.patch_size),
        transform=None,
        num_frames=1,
        data_scale=args.data_scale,
        repeats_per_frame=1,
    )
    test_dataset = LorealSequenceDataset(
        sequence_info=test_seq,
        patch_size=None,
        transform=None,
        num_frames=1,
        data_scale=args.data_scale,
        repeats_per_frame=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    n_train_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    print(
        f"Train items: {len(train_dataset)} (~{n_train_batches} batches/epoch, "
        f"repeats_per_frame={args.repeats_per_frame})"
    )
    print(f"Val items: {len(val_dataset)} / Test items: {len(test_dataset)}")

    # 4) Model and objective
    model = dinv.models.ArtifactRemoval(
        dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained=None, nc=[16, 32, 64, 128])
    ).to(device) #pretrained=None es para que no descargue los pesos de huggingface

    if args.pretrained_ckpt: #Este es mi checkpoint, no es el de huggingface
        ckpt_path = Path(args.pretrained_ckpt)
        if ckpt_path.exists():
            print(f"Loading pre-trained weights from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    cleaned[k[6:]] = v
                elif not k.startswith("noise_model."):
                    cleaned[k] = v
            model.load_state_dict(cleaned, strict=False)
        else:
            print(f"WARNING: pretrained_ckpt not found: {ckpt_path}")

    if args.loss != "gr2r_mse":
        raise ValueError("Only gr2r_mse is supported in this script version")

    criterion = R2RLoss(noise_model=noise_model, alpha=args.alpha)
    model = criterion.adapt_model(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 5) Train with validation by val_loss
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    best_epoch = -1
    best_ckpt_path = output_dir / "best_model.pth"

    for epoch in range(args.epochs):
        model.train()
        running_train = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for y_stack, _ in pbar:
            y_stack = y_stack.to(device)
            y = _to_model_input(y_stack)

            optimizer.zero_grad()
            x_est = model(y, physics, update_parameters=True)
            loss = criterion(x_est, y, physics, model)
            loss.backward()
            optimizer.step()

            running_train += loss.item()
            pbar.set_postfix({"train_loss": f"{loss.item():.5f}"})

        train_loss_epoch = running_train / max(len(train_loader), 1)
        val_loss_epoch = evaluate_val_loss(model, val_loader, criterion, physics, args.eval_seed)

        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss_epoch)
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss_epoch:.6f}, "
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
            periodic_path = output_dir / f"model_epoch{epoch + 1}.pth"
            torch.save(model.state_dict(), periodic_path)
            print(f"Periodic checkpoint saved: {periodic_path}")

    # 6) Save loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loreal real-sequences training/validation loss")
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

    # 7) Export selected sequences
    explicit_viz = [s for s in val_seq + test_seq if Path(s[0]).name in set(visualize_names)]
    if len(explicit_viz) > 0:
        export_sequences(model, physics, explicit_viz, output_dir, tag="viz", max_sequences=args.max_export_sequences)
    else:
        export_sequences(model, physics, val_seq, output_dir, tag="val", max_sequences=args.max_export_sequences)
        export_sequences(model, physics, test_seq, output_dir, tag="test", max_sequences=args.max_export_sequences)

    print(f"Finished. Check results in {output_dir}")


class Args:
    loss = "gr2r_mse"
    noise = 1 / 255.0
    alpha = 0.15

    epochs = 40
    batch_size = 16
    val_batch_size = 16
    lr = 1e-4
    patch_size = 256
    data_scale = 255.0
    num_workers = 4
    repeats_per_frame = 10
    eval_seed = 43
    checkpoint_every = 10

    # Optional transfer learning from FMDD
    pretrained_ckpt = "results/denoising-poisson-fmdd-drunet/tif_output_2026_06_02-19_42_17/best_model.pth"

    # Used only when loreal_split.txt is missing
    val_prefixes = ["HF1_", "Mela1_"]
    test_prefixes = ["HF2_", "Mela2_"]

    max_export_sequences = 2


if __name__ == "__main__":
    args = Args()
    train_model(args)
