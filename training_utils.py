"""
Shared utilities for GR2R training and inference scripts.

Instead of copying these classes/functions into every script, they live here
and each script imports what it needs.
"""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn


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


def save_parameters(args, output_dir, script_name, device):
    """Save experiment hyperparameters to parameters.txt inside output_dir.

    Args:
        args:        The Args object with all hyperparameters.
        output_dir:  Path to the experiment output folder.
        script_name: Name of the calling script (use Path(__file__).name).
        device:      The torch device being used (e.g. 'cuda:0' or 'cpu').
    """
    with open(output_dir / "parameters.txt", "w") as f:
        f.write(f"Experiment: {script_name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write("-" * 50 + "\n")
        for key in dir(args):
            if not key.startswith("_"):
                value = getattr(args, key)
                if not callable(value):
                    f.write(f"{key} = {value}\n")
    print(f"Parameters saved to {output_dir / 'parameters.txt'}")


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
