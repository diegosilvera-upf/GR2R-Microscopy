r"""
Unified GR2R training script. All hyperparameters are defined in a YAML config file.

Usage:
  python train.py --config configs/fmdd_drunet.yaml
  python train.py --config configs/loreal_fastdvdnet.yaml
"""

from __future__ import annotations

import argparse
import sys
sys.path.insert(0, "/home/diegosilvera/Escritorio/learning2recorrupt")
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import deepinv as dinv
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.optim as optim
import yaml
from deepinv.loss import PSNR, R2RLoss
from l2r import L2RLoss, Recorruptor
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import (
    FMDDataset,
    LorealSequenceDataset,
    get_fmdd_sequences,
    get_fmdd_split_from_file,
    get_valid_sequences,
)
from models_FastDVDnet_sans_noise_map import FastDVDnet
from training_utils import (
    FastDVDNetContextWrapper,
    build_eval_cache,
    load_loreal_split,
    save_parameters,
)

BASE_DIR = Path(".")
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(yaml_path: str) -> SimpleNamespace:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    defaults = {
        "pretrained_ckpt": None,
        "inference_dir": None,
        "n_eval_sequences": None,
        "checkpoint_every": 10,
        "max_export_sequences": 3,
        "num_workers": 4,
        "repeats_per_sequence": 55,
        "repeats_per_frame": 10,
        "val_batch_size": None,
        "val_prefixes": [],
        "test_prefixes": [],
        "fmdd_data_dir": "../data/FMDD",
        "fmdd_split_file": "txts/fmdd_split.txt",
        "fmdd_modalities": None,
        "loreal_data_dir": None,
        "loreal_split_file": "txts/loreal_split.txt",
        "data_scale": None,
        "eval_seed": 42,
        "loss": "r2r",
        "l2r_eval_n_samples": 5,
        "l2r_recorruptor_ckpt": None,
        "l2r_recorruptor_lr": 1e-4,
    }
    for k, v in defaults.items():
        raw.setdefault(k, v)

    if raw["data_scale"] is None:
        raw["data_scale"] = 1.0 if raw["dataset"] == "fmdd" else 255.0
    if raw["val_batch_size"] is None:
        raw["val_batch_size"] = raw["batch_size"]

    return SimpleNamespace(**raw)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(cfg: SimpleNamespace, noise_model) -> torch.nn.Module:
    if cfg.loss == "l2r":
        recorruptor = Recorruptor(kernel_size=1, multiplicative=True, sigma=0.4).to(device)
        if cfg.l2r_recorruptor_ckpt:
            recorruptor.load_state_dict(torch.load(cfg.l2r_recorruptor_ckpt, map_location=device))
        criterion = L2RLoss(recorruptor=recorruptor, alpha=cfg.alpha, eval_n_samples=cfg.l2r_eval_n_samples, recorruptor_lr=cfg.l2r_recorruptor_lr)
    else:
        criterion = R2RLoss(noise_model=noise_model, alpha=cfg.alpha)
    if cfg.model == "drunet":
        backbone = dinv.models.DRUNet(
            in_channels=1, out_channels=1, pretrained=None, nc=[16, 32, 64, 128]
        )
        model = dinv.models.ArtifactRemoval(backbone).to(device)
    else:
        base = FastDVDnet(num_input_frames=5).to(device)
        model = FastDVDNetContextWrapper(base).to(device)
    return criterion.adapt_model(model), criterion


def load_pretrained(model: torch.nn.Module, cfg: SimpleNamespace) -> None:
    if not cfg.pretrained_ckpt:
        return
    ckpt_path = Path(cfg.pretrained_ckpt)
    if not ckpt_path.exists():
        print(f"WARNING: pretrained_ckpt not found: {ckpt_path}")
        return
    print(f"Loading pretrained weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    if cfg.model == "drunet":
        # R2RModel wraps the backbone under "model." — strip that prefix so keys match.
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned[k[6:]] = v
            elif not k.startswith("noise_model."):
                cleaned[k] = v
    else:
        cleaned = {k: v for k, v in state_dict.items() if not k.startswith("noise_model.")}

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  {len(missing)} missing keys")
    if unexpected:
        print(f"  {len(unexpected)} unexpected keys")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def build_fmdd_datasets(cfg: SimpleNamespace):
    data_dir = Path(cfg.fmdd_data_dir)
    sequences = get_fmdd_sequences(data_dir, modalities=cfg.fmdd_modalities)
    print(f"Found {len(sequences)} FMDD sequences.")

    split_file = Path(cfg.fmdd_split_file)
    train_seq, test_seq, visualize_indices = get_fmdd_split_from_file(sequences, split_file)
    print(f"Split: {len(train_seq)} train / {len(test_seq)} test")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    num_frames = 5 if cfg.model == "fastdvdnet" else 1

    train_dataset = FMDDataset(
        sequence_info=train_seq,
        patch_size=(cfg.patch_size, cfg.patch_size),
        mode="clean",
        data_scale=cfg.data_scale,
        num_frames=num_frames,
        transform=transform,
        repeats_per_sequence=cfg.repeats_per_sequence,
    )
    test_dataset = FMDDataset(
        sequence_info=test_seq,
        mode="clean",
        data_scale=cfg.data_scale,
        num_frames=num_frames,
        repeats_per_sequence=1,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    return train_loader, train_dataset, test_dataset, test_seq, visualize_indices


def build_loreal_datasets(cfg: SimpleNamespace):
    data_dir = Path(cfg.loreal_data_dir)
    sequence_paths = sorted(data_dir.glob("*"))
    valid_sequences = get_valid_sequences(sequence_paths)
    print(f"Found {len(valid_sequences)} valid Loreal sequences.")

    train_seq, val_seq, test_seq, visualize_names = load_loreal_split(
        valid_sequences=valid_sequences,
        split_file=Path(cfg.loreal_split_file),
        val_prefixes=cfg.val_prefixes,
        test_prefixes=cfg.test_prefixes,
    )
    print(f"Split: {len(train_seq)} train / {len(val_seq)} val / {len(test_seq)} test")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    num_frames = 5 if cfg.model == "fastdvdnet" else 1

    train_dataset = LorealSequenceDataset(
        sequence_info=train_seq,
        patch_size=(cfg.patch_size, cfg.patch_size),
        transform=transform,
        num_frames=num_frames,
        data_scale=cfg.data_scale,
        repeats_per_frame=cfg.repeats_per_frame,
    )
    val_dataset = LorealSequenceDataset(
        sequence_info=val_seq,
        patch_size=(cfg.patch_size, cfg.patch_size),
        transform=None,
        num_frames=num_frames,
        data_scale=cfg.data_scale,
        repeats_per_frame=1,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    n_train_batches = (len(train_dataset) + cfg.batch_size - 1) // cfg.batch_size
    print(
        f"Train: {len(train_dataset)} items (~{n_train_batches} batches/epoch) | "
        f"Val: {len(val_dataset)} items"
    )
    return train_loader, val_loader, train_seq, val_seq, test_seq, visualize_names


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_fmdd(model, cfg, physics, criterion, eval_cache) -> tuple[float, float]:
    """Returns (mean_psnr, mean_val_loss) on the fixed eval cache."""
    model.eval()
    psnr_sum = 0.0
    val_loss_sum = 0.0
    n = len(eval_cache)

    for i, (x_gt_cpu, y_noisy_cpu) in enumerate(eval_cache):
        x_gt = x_gt_cpu.to(device)
        y_noisy = y_noisy_cpu.to(device)

        if cfg.model == "fastdvdnet":
            y_central = y_noisy[:, 2:3]
            with torch.no_grad():
                model.model.set_context(y_noisy)
                x_est = model(y_central, physics)
                psnr_sum += PSNR()(x=x_gt, x_net=x_est).item()

            torch.manual_seed(cfg.eval_seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.eval_seed + i)
            model.model.set_context(y_noisy)
            model.training = True
            x_est_loss = model(y_central, physics, update_parameters=True)
            model.training = False
            val_loss_sum += criterion(x_est_loss, y_central, physics, model).item()
        else:
            with torch.no_grad():
                x_est = model(y_noisy, physics)
                psnr_sum += PSNR()(x=x_gt, x_net=x_est).item()

            torch.manual_seed(cfg.eval_seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.eval_seed + i)
            model.training = True
            x_est_loss = model(y_noisy, physics, update_parameters=True)
            model.training = False
            val_loss_sum += criterion(x_est_loss, y_noisy, physics, model).item()

    return psnr_sum / n, val_loss_sum / n


def evaluate_loreal(model, cfg, val_loader, physics, criterion) -> float:
    """Returns mean val_loss over the validation set."""
    model.eval()
    running = 0.0
    n_batches = 0

    for batch_idx, (y_stack, _) in enumerate(val_loader):
        y_stack = y_stack.to(device)

        torch.manual_seed(cfg.eval_seed + batch_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.eval_seed + batch_idx)

        if cfg.model == "fastdvdnet":
            y_central = y_stack[:, 2:3]
            model.model.set_context(y_stack)
            model.training = True
            x_est = model(y_central, physics, update_parameters=True)
            model.training = False
            loss_val = criterion(x_est, y_central, physics, model)
        else:
            y = y_stack[:, 0:1]
            model.training = True
            x_est = model(y, physics, update_parameters=True)
            model.training = False
            loss_val = criterion(x_est, y, physics, model)

        running += loss_val.item()
        n_batches += 1

    return running / n_batches if n_batches > 0 else float("inf")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_fmdd_samples(model, cfg, physics, test_dataset, visualize_indices, output_dir, tag=""):
    export_indices = visualize_indices if visualize_indices else [0, 1, 2]
    model.eval()
    suffix = f"_{tag}" if tag else ""
    with torch.no_grad():
        for i in export_indices:
            if i >= len(test_dataset):
                break
            x_clean_stack, x_clean = test_dataset[i]
            x_clean_stack = x_clean_stack.unsqueeze(0).to(device)
            x_gt = x_clean.unsqueeze(0).to(device)

            torch.manual_seed(cfg.eval_seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.eval_seed + i)

            if cfg.model == "fastdvdnet":
                noisy_frames = [physics(x_clean_stack[:, j:j+1]) for j in range(5)]
                stack_noisy = torch.cat(noisy_frames, dim=1)
                y_central = stack_noisy[:, 2:3]
                model.model.set_context(stack_noisy)
                x_est = model(y_central, physics)
                y_for_save = y_central
            else:
                y_for_save = physics(x_clean_stack[:, 0:1])
                x_est = model(y_for_save, physics)

            tifffile.imwrite(str(output_dir / f"fmd_{i}{suffix}_denoised.tif"), x_est.squeeze().cpu().numpy().astype(np.float32))
            tifffile.imwrite(str(output_dir / f"fmd_{i}{suffix}_noisy.tif"), y_for_save.squeeze().cpu().numpy().astype(np.float32))
            tifffile.imwrite(str(output_dir / f"fmd_{i}{suffix}_clean.tif"), x_gt.squeeze().cpu().numpy().astype(np.float32))
    print(f"Exported {len(export_indices)} FMDD samples{suffix}.")


def export_loreal_sequences(model, cfg, physics, sequences, output_dir, tag=""):
    if not sequences:
        print(f"No sequences to export for tag={tag}.")
        return
    chosen = sequences[: cfg.max_export_sequences]
    print(f"Exporting {len(chosen)} Loreal sequence(s) (tag={tag or 'default'}).")
    suffix = f"_{tag}" if tag else ""
    num_frames = 5 if cfg.model == "fastdvdnet" else 1

    model.eval()
    with torch.no_grad():
        for seq_path, a, b in chosen:
            seq_name = Path(seq_path).name
            seq_ds = LorealSequenceDataset(
                sequence_info=[(seq_path, a, b)],
                num_frames=num_frames,
                data_scale=cfg.data_scale,
            )
            if len(seq_ds) == 0:
                print(f"  SKIP {seq_name}: no valid stacks.")
                continue

            denoised_frames = []
            noisy_frames = []
            for i in tqdm(range(len(seq_ds)), desc=f"{tag}:{seq_name}", leave=False):
                y_stack, y_target = seq_ds[i]
                y_stack = y_stack.unsqueeze(0).to(device)
                y_target = y_target.unsqueeze(0).to(device)

                if cfg.model == "fastdvdnet":
                    model.model.set_context(y_stack)
                    x_est = model(y_target, physics)
                else:
                    x_est = model(y_target, physics)

                # denoised_frames.append(x_est.squeeze().cpu().numpy().astype(np.float32))
                # noisy_frames.append(y_target.squeeze().cpu().numpy().astype(np.float32))
                # después
                denoised_frames.append((x_est * cfg.data_scale).squeeze().cpu().numpy().astype(np.float32))
                noisy_frames.append((y_target * cfg.data_scale).squeeze().cpu().numpy().astype(np.float32))

            safe_name = seq_name.replace("/", "_")
            tifffile.imwrite(str(output_dir / f"{suffix}{safe_name}_denoised.tif"), np.stack(denoised_frames, axis=0))
            tifffile.imwrite(str(output_dir / f"{suffix}{safe_name}_noisy.tif"), np.stack(noisy_frames, axis=0))
            print(f"  Saved {safe_name} ({len(denoised_frames)} frames).")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def save_config(cfg: SimpleNamespace, output_dir: Path, config_path: str) -> None:
    import shutil
    shutil.copy(config_path, output_dir / "config.yaml")
    # Also write parameters.txt so inference.py --use-params-from-checkpoint-dir works.
    with open(output_dir / "parameters.txt", "w") as f:
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write("-" * 50 + "\n")
        for k, v in vars(cfg).items():
            f.write(f"{k} = {v}\n")
    print(f"Config saved to {output_dir / 'config.yaml'}")


def save_loss_plot(train_losses, val_losses, val_psnrs, cfg, output_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, label="Train Loss")
    ax1.plot(epochs_range, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    if val_psnrs:
        ax2 = ax1.twinx()
        ax2.plot(epochs_range, val_psnrs, "g--", label="Val PSNR (dB)")
        ax2.set_ylabel("Val PSNR (dB)")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
    else:
        ax1.legend()

    title = f"{cfg.project_name} — {cfg.model} / {cfg.dataset}"
    ax1.set_title(title)
    path = output_dir / "loss_plot.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Loss plot saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GR2R unified training.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.model in ("drunet", "fastdvdnet"), f"Unknown model: {cfg.model}"
    assert cfg.dataset in ("fmdd", "loreal"), f"Unknown dataset: {cfg.dataset}"
    assert cfg.loss in ("r2r", "l2r"), f"Unknown loss: {cfg.loss}"

    # Output directory
    if cfg.inference_dir is not None:
        output_dir = Path(cfg.inference_dir)
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        results_dir = BASE_DIR / "results" / cfg.project_name
        results_dir.mkdir(parents=True, exist_ok=True)
        output_dir = results_dir / f"tif_output_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir, args.config)

    print(f"Device: {device} | model: {cfg.model} | dataset: {cfg.dataset}")
    print(f"Output: {output_dir}")

    # Physics
    noise_model = dinv.physics.PoissonNoise(cfg.gamma)
    noise_model.sigma = cfg.gamma
    physics = dinv.physics.Denoising(noise_model=noise_model)

    # Model
    model, criterion = build_model(cfg, noise_model)
    load_pretrained(model, cfg)
    if cfg.loss == "l2r":
        optimizer = optim.Adam(model.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.epochs * 0.8) + 1)

    # If inference_dir is set: skip training, just export
    if cfg.inference_dir is not None:
        print("inference_dir set — skipping training, running export only.")
        if cfg.dataset == "fmdd":
            _, _, test_dataset, test_seq, visualize_indices = build_fmdd_datasets(cfg)
            export_fmdd_samples(model, cfg, physics, test_dataset, visualize_indices, output_dir)
        else:
            _, _, _, val_seq, test_seq, visualize_names = build_loreal_datasets(cfg)
            explicit_viz = [s for s in val_seq + test_seq if Path(s[0]).name in set(visualize_names)]
            seqs = explicit_viz or val_seq
            export_loreal_sequences(model, cfg, physics, seqs, output_dir)
        print(f"Done. Outputs in {output_dir}")
        return

    # Datasets
    num_frames = 5 if cfg.model == "fastdvdnet" else 1

    if cfg.dataset == "fmdd":
        train_loader, train_dataset, test_dataset, test_seq, visualize_indices = build_fmdd_datasets(cfg)
        n_eval = len(test_seq)
        if cfg.n_eval_sequences is not None:
            n_eval = min(n_eval, cfg.n_eval_sequences)
        eval_cache = build_eval_cache(test_dataset, physics, n_eval, device, cfg.eval_seed, num_frames=num_frames)
        val_loader = None
        val_seq = None
        visualize_names = None
    else:
        train_loader, val_loader, train_seq, val_seq, test_seq, visualize_names = build_loreal_datasets(cfg)
        eval_cache = None
        test_dataset = None
        visualize_indices = None

    # Checkpoint tracking — FMDD keeps best-by-loss AND best-by-PSNR; Loreal only best-by-loss
    best_val_loss = float("inf")
    best_loss_epoch = -1
    best_ckpt_loss = output_dir / "best_model_loss.pth"

    best_psnr = 0.0
    best_psnr_epoch = -1
    best_ckpt_psnr = output_dir / "best_model_psnr.pth"  # only used for FMDD

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_psnrs: list[float] = []  # empty for Loreal

    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        running_train = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            # Prepare inputs
            if cfg.dataset == "fmdd":
                x_clean_stack, _ = batch
                x_clean_stack = x_clean_stack.to(device)
                if cfg.model == "drunet":
                    y = physics(x_clean_stack[:, 0:1])
                else:
                    noisy_frames = [physics(x_clean_stack[:, j:j+1]) for j in range(5)]
                    y = torch.cat(noisy_frames, dim=1)
            else:  # loreal — data already noisy
                y_stack, _ = batch
                y = y_stack.to(device)

            # Forward pass
            if cfg.model == "drunet":
                x_est = model(y, physics, update_parameters=True)
                loss = criterion(x_est, y, physics, model)
            else:
                y_central = y[:, 2:3]
                model.model.set_context(y)
                x_est = model(y_central, physics, update_parameters=True)
                loss = criterion(x_est, y_central, physics, model)

            loss.backward()
            optimizer.step()
            running_train += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        train_loss_epoch = running_train / max(len(train_loader), 1)

        # Evaluation
        if cfg.dataset == "fmdd":
            current_psnr, current_val_loss = evaluate_fmdd(model, cfg, physics, criterion, eval_cache)
            val_psnrs.append(current_psnr)
            print(
                f"Epoch {epoch+1}: train_loss={train_loss_epoch:.6f} | "
                f"val_loss={current_val_loss:.6f} | val_psnr={current_psnr:.2f} dB"
            )
        else:
            current_val_loss = evaluate_loreal(model, cfg, val_loader, physics, criterion)
            current_psnr = None
            print(
                f"Epoch {epoch+1}: train_loss={train_loss_epoch:.6f} | "
                f"val_loss={current_val_loss:.6f}"
            )

        train_losses.append(train_loss_epoch)
        val_losses.append(current_val_loss)

        # Checkpoint by val_loss
        is_best = (
            abs(current_val_loss) < abs(best_val_loss)
            if cfg.loss == "l2r"
            else current_val_loss < best_val_loss
        )
        if is_best:
            best_val_loss = current_val_loss
            best_loss_epoch = epoch + 1
            torch.save(model.state_dict(), best_ckpt_loss)
            print(f"  New best val_loss model saved (epoch={best_loss_epoch})")

        # Checkpoint by PSNR (FMDD only)
        if current_psnr is not None and current_psnr > best_psnr:
            best_psnr = current_psnr
            best_psnr_epoch = epoch + 1
            torch.save(model.state_dict(), best_ckpt_psnr)
            print(f"  New best PSNR model saved (epoch={best_psnr_epoch}, {best_psnr:.2f} dB)")

        scheduler.step()

        # Periodic checkpoint
        if (epoch + 1) % cfg.checkpoint_every == 0:
            periodic_path = output_dir / f"model_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), periodic_path)
            print(f"  Periodic checkpoint: {periodic_path}")

    # Save training summary
    with open(output_dir / "best_checkpoint.txt", "w") as f:
        f.write(f"best_loss_epoch={best_loss_epoch}\n")
        f.write(f"best_val_loss={best_val_loss:.6f}\n")
        f.write(f"weights_loss_path={best_ckpt_loss}\n")
        if cfg.dataset == "fmdd":
            f.write(f"best_psnr_epoch={best_psnr_epoch}\n")
            f.write(f"best_val_psnr={best_psnr:.4f}\n")
            f.write(f"weights_psnr_path={best_ckpt_psnr}\n")

    save_loss_plot(train_losses, val_losses, val_psnrs if val_psnrs else None, cfg, output_dir)

    # Export results with best weights
    if cfg.dataset == "fmdd":
        for ckpt_path, epoch_n, tag in [
            (best_ckpt_psnr, best_psnr_epoch, "psnr"),
            (best_ckpt_loss, best_loss_epoch, "loss"),
        ]:
            if ckpt_path.exists():
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"Exporting with best_{tag} weights (epoch {epoch_n})...")
                export_fmdd_samples(model, cfg, physics, test_dataset, visualize_indices, output_dir, tag=tag)
    else:
        if best_ckpt_loss.exists():
            model.load_state_dict(torch.load(best_ckpt_loss, map_location=device))
        explicit_viz = [s for s in val_seq + test_seq if Path(s[0]).name in set(visualize_names)]
        if explicit_viz:
            export_loreal_sequences(model, cfg, physics, explicit_viz, output_dir, tag="viz")
        else:
            export_loreal_sequences(model, cfg, physics, val_seq, output_dir, tag="val")
            export_loreal_sequences(model, cfg, physics, test_seq, output_dir, tag="test")

    print(f"Finished. Check results in {output_dir}")


if __name__ == "__main__":
    main()
