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
from training_utils import FastDVDNetContextWrapper, save_parameters, load_loreal_split

# from loreal_dataset import get_valid_sequences, LorealSequenceDataset
from dataset import get_valid_sequences, LorealSequenceDataset
import matplotlib.pyplot as plt

from models_FastDVDnet_sans_noise_map import FastDVDnet

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
    save_parameters(args, output_dir, script_name=Path(__file__).name, device=device)

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
    alpha = 0.85
    epochs = 3
    batch_size = 32
    val_batch_size = 32
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
