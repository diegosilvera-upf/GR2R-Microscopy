"""
Shared utilities for GR2R training and inference scripts.

Instead of copying these classes/functions into every script, they live here
and each script imports what it needs.
"""

import sys
sys.path.insert(0, "/home/diegosilvera/Escritorio/learning2recorrupt")
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import deepinv as dinv
import torch
import torch.nn as nn
from deepinv.loss import PSNR, R2RLoss
from l2r import L2RLoss, Recorruptor
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import (
    FMDDataset,
    LorealSequenceDataset,
    get_fmdd_sequences,
    get_fmdd_split_from_file,
    get_valid_sequences,
)
from models_FastDVDnet_sans_noise_map import FastDVDnet

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


class FastDVDNetContextWrapper(nn.Module):
    """Wraps FastDVDnet to manage the 5-frame context for R2RLoss.

    FastDVDnet expects a 5-frame stack as input. During training with R2RLoss,
    deepinv recorrupts only the central frame (position 2). This wrapper keeps
    the other 4 frames at their original noise level and replaces only the
    central one with whatever R2RLoss passes in.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._context = None

    def set_context(self, stack):
        """Store the full 5-frame stack before each forward pass."""
        self._context = stack.detach()

    def forward(self, y_central, physics=None, **kwargs):
        if self._context is None:
            raise RuntimeError("Call set_context(stack) before forward pass.")
        stack = self._context.clone()
        stack[:, 2:3, :, :] = y_central
        return self.model(stack)


def load_loreal_split(valid_sequences, split_file, val_prefixes=None, test_prefixes=None):
    """Split Loreal sequences into train / val / test sets.

    Reads a CSV-style split file where each line is:
        sequence_folder_name, train|val|test [, True]
    The optional third column marks sequences for visualization export.

    If the split file does not exist, falls back to a prefix-based split
    using val_prefixes and test_prefixes, or a 90/10 train/val split if
    no prefixes match.

    Returns:
        train_seq, val_seq, test_seq  — lists of (seq_path, a, b) tuples
        visualize_names               — list of sequence names flagged for export
    """
    seq_by_name = {
        Path(seq_path).name: (seq_path, a, b)
        for seq_path, a, b in valid_sequences
    }

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
                    print(f"  WARNING: {seq_name} in split file not found in valid sequences.")
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
        leftovers = [
            item for item in valid_sequences
            if Path(item[0]).name not in assigned
        ]
        train_seq.extend(leftovers)
        if leftovers:
            print(f"Added {len(leftovers)} unassigned sequences to train.")

    else:
        print("No split file found. Using prefix fallback split.")
        val_prefixes = val_prefixes or []
        test_prefixes = test_prefixes or []

        def starts_with_any(name, prefixes):
            return any(name.startswith(p) for p in prefixes)

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
            print("Fallback had no val matches — using deterministic 90/10 train/val split.")

    return train_seq, val_seq, test_seq, visualize_names


def build_eval_cache(test_dataset, physics, n_eval, device, seed, num_frames=1):
    """Pre-generate fixed (x_gt, y_noisy) pairs for reproducible evaluation.

    Without a fixed cache, Poisson noise is resampled every epoch so metrics
    are not comparable across epochs.

    Args:
        num_frames: 1 for DRUNet (noise on one frame), >1 for FastDVDNet
                    (noise applied independently to each of the num_frames frames).
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

            if num_frames == 1:
                noisy = physics(x_clean_stack[:, 0:1, :, :])
            else:
                frames = [physics(x_clean_stack[:, j:j+1]) for j in range(num_frames)]
                noisy = torch.cat(frames, dim=1)

            cache.append((x_gt.detach().cpu(), noisy.detach().cpu()))

    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    print(f"Built eval cache with {len(cache)} fixed noisy samples (seed={seed}).")
    return cache


def build_train_model(cfg: SimpleNamespace, noise_model) -> torch.nn.Module:
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
