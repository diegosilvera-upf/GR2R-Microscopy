"""
Dataset classes and sequence-discovery utilities for GR2R training and inference.

Loreal section:
  get_valid_sequences    — Filter and discover valid Loreal sequence directories.
  LorealSequenceDataset  — Multi-frame (or single-frame) dataset from real Loreal sequences.

FMDD section:
  get_fmdd_sequences       — Discover sequences in the FMDD folder structure.
  get_fmdd_split_from_file — Load train/test split from a .txt file.
  FMDDataset               — FMDD dataset supporting 'clean', 'synthetic', and 'raw' modes.
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import imageio.v3 as iio


# ==============================================================================
# Loreal
# ==============================================================================


def get_valid_sequences(
    sequence_paths,
    out_file_invalid="sequences_left_out.txt",
    out_file_valid="sequences_used.txt",
):
    """
    Filter Loreal sequence directories, keeping only those that:
      - have a pre-processing.txt with gain value a close to 1 (|a-1| <= 0.2)
      - contain at least one channel with >= 5 .tif frames

    Writes rejected sequences to out_file_invalid and accepted ones to out_file_valid.
    Returns a sorted list of (seq_path, a, b) tuples.
    """
    valid_sequences = []
    with open(out_file_invalid, "w") as f_invalid, open(out_file_valid, "w") as f_valid:
        for seq in sequence_paths:
            seq = Path(seq)
            if not seq.is_dir():
                continue

            preproc_file = seq / "pre-processing.txt"
            if not preproc_file.exists():
                f_invalid.write(f"{seq.name}, no pre-processing.txt file\n")
                continue

            try:
                params = np.loadtxt(preproc_file)
                if params.ndim == 1:
                    a, b = params[0], params[1]
                else:
                    a, b = params.flatten()[0], params.flatten()[1]
            except Exception as e:
                f_invalid.write(f"{seq.name}, error reading pre-processing.txt: {e}\n")
                continue

            if np.abs(a - 1) > 0.2:
                f_invalid.write(f"{seq.name}, a={a}\n")
                continue

            tif_files = sorted(seq.glob("*.tif"))
            if not tif_files:
                continue

            names = [f.name for f in tif_files]
            channels = (
                ["_c0_", "_c1_"]
                if any("_c0_" in n or "_c1_" in n for n in names)
                else [""]
            )

            has_enough_frames = False
            for ch in channels:
                frames = [f for f in tif_files if ch in f.name] if ch else tif_files
                if len(frames) >= 5:
                    has_enough_frames = True
                    break

            if has_enough_frames:
                valid_sequences.append((str(seq), float(a), float(b)))
                f_valid.write(f"{seq.name}, a={a}, b={b}\n")
            else:
                f_invalid.write(f"{seq.name}, not enough frames (min 5)\n")

    return sorted(valid_sequences)


class LorealSequenceDataset(Dataset):
    """
    Dataset for real Loreal sequences, supporting single-frame (DRUNet) and
    multi-frame (FastDVDNet) stacks.

    Args:
        sequence_info:     List of (seq_path, a, b) tuples from get_valid_sequences.
        patch_size:        (H, W) tuple for random cropping, or None for full frames.
        transform:         Optional torchvision transform applied to the full stack.
        data_scale:        Divide raw pixel values by this. Use 255.0 for Loreal.
        num_frames:        Frames per stack. 1 for DRUNet, 5 for FastDVDNet.
        repeats_per_frame: How many times each frame position is listed per epoch.
                           Higher values = more random patches sampled per sequence.
    """

    def __init__(
        self,
        sequence_info,
        patch_size=None,
        transform=None,
        data_scale=255.0,
        num_frames=5,
        repeats_per_frame=1,
    ):
        self.patch_size = patch_size
        self.transform = transform
        self.data_scale = data_scale
        self.num_frames = num_frames
        self.repeats_per_frame = max(1, int(repeats_per_frame))
        self.stacks = []

        for seq_path, a, b in sequence_info:
            seq = Path(seq_path)
            tif_files = sorted(seq.glob("*.tif"))
            names = [f.name for f in tif_files]
            channels = (
                ["_c0_", "_c1_"]
                if any("_c0_" in n or "_c1_" in n for n in names)
                else [""]
            )

            for ch in channels:
                frames = (
                    sorted(f for f in tif_files if ch in f.name)
                    if ch
                    else sorted(tif_files)
                )
                if len(frames) < self.num_frames:
                    continue

                mid = self.num_frames // 2
                for i in range(mid, len(frames) - mid):
                    stack_paths = [
                        str(f) for f in frames[i - mid : i - mid + self.num_frames]
                    ]
                    for _ in range(self.repeats_per_frame):
                        self.stacks.append((stack_paths, float(a), float(b)))

    def __len__(self):
        return len(self.stacks)

    def _read_tif(self, path):
        img = iio.imread(str(path)).astype(np.float32)
        img = torch.from_numpy(img)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3:
            img = img.permute(2, 0, 1).mean(dim=0, keepdim=True)
        return img  # [1, H, W]

    def make_divisible_by_4(self, img):
        H, W = img.shape[-2:]
        return img[..., : (H // 4) * 4, : (W // 4) * 4]

    def __getitem__(self, idx):
        stack_paths, a, b = self.stacks[idx]
        frames = [self._read_tif(p) for p in stack_paths]
        stack = torch.cat(frames, dim=0)  # [num_frames, H, W]

        stack = self.make_divisible_by_4(stack)
        stack = stack / self.data_scale
        stack = torch.clamp(stack, min=0.0)

        if self.patch_size is not None:
            H, W = stack.shape[-2:]
            ph, pw = self.patch_size
            if H >= ph and W >= pw:
                top = torch.randint(0, H - ph + 1, (1,)).item()
                left = torch.randint(0, W - pw + 1, (1,)).item()
                stack = stack[:, top : top + ph, left : left + pw]

        if self.transform:
            stack = self.transform(stack)

        target = stack[self.num_frames // 2 : self.num_frames // 2 + 1].clone()  # [1, H, W]
        return stack, target


# ==============================================================================
# FMDD
# ==============================================================================


def get_fmdd_sequences(root_dir, modalities=None):
    """
    Discover sequences in the FMDD directory structure:
      root_dir / Modality / raw / SequenceID / *.png

    Returns a list of dicts with keys: 'modality', 'seq_id', 'frames', 'gt'.
    Order is deterministic (sorted by modality then seq_id).
    """
    root = Path(root_dir)
    sequences = []

    if modalities is None:
        modalities = sorted(d.name for d in root.iterdir() if d.is_dir())
        modalities = [m for m in modalities if (root / m / "raw").exists()]

    for mod in modalities:
        mod_raw = root / mod / "raw"
        if not mod_raw.exists():
            continue

        for seq_dir in sorted(mod_raw.iterdir()):
            if not seq_dir.is_dir():
                continue

            png_files = sorted(seq_dir.glob("*.png"))
            if len(png_files) < 5:
                continue

            gt_path = root / mod / "gt" / seq_dir.name / "avg50.png"
            sequences.append({
                "modality": mod,
                "seq_id": seq_dir.name,
                "frames": [str(p) for p in png_files],
                "gt": str(gt_path) if gt_path.exists() else None,
            })

    return sequences


def get_fmdd_split_from_file(sequences, split_file):
    """
    Load a train/test split from a .txt file.

    Each non-comment line must be:  Modality/seq_id, train|test [, True]
    The optional third column marks a sequence for visualization export.

    Returns:
        train_seqs, test_seqs  — lists of sequence dicts
        visualize_indices      — list of integer positions within test_seqs
    """
    split_info = {}
    viz_keys = []

    with open(split_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                split_info[parts[0]] = parts[1]
            if len(parts) >= 3 and parts[2].lower() == "true":
                viz_keys.append(parts[0])

    train_seqs, test_seqs = [], []
    for s in sequences:
        key = f"{s['modality']}/{s['seq_id']}"
        if split_info.get(key, "train") in ("test", "val"):
            test_seqs.append(s)
        else:
            train_seqs.append(s)

    visualize_indices = [
        i
        for i, s in enumerate(test_seqs)
        if f"{s['modality']}/{s['seq_id']}" in viz_keys
    ]

    return train_seqs, test_seqs, visualize_indices


class FMDDataset(Dataset):
    """
    Dataset for FMDD (Fluorescence Microscopy Denoising Dataset).

    Three modes:
      'clean'     — Returns GT image only; noise is added externally by physics/deepinv.
                    This is the mode to use with R2RLoss (GR2R training).
      'synthetic' — Adds Poisson noise internally (requires gamma). Use for sanity checks.
      'raw'       — Uses the real noisy PNG frames from the FMDD measurements folder.

    Note on normalization: _read_png already maps pixel values to [0, 1] via /255.
    data_scale is applied on top of that, so leave it at 1.0 for FMDD.

    Args:
        sequence_info:        List of sequence dicts from get_fmdd_sequences.
        patch_size:           (H, W) random crop size, or None for full images.
        transform:            Optional transform applied jointly to stack and target.
        data_scale:           Extra scaling after /255 normalization. Keep at 1.0 for FMDD.
        mode:                 'clean', 'synthetic', or 'raw'.
        gamma:                Poisson gain for mode='synthetic'. Ignored otherwise.
        num_frames:           Frames per stack. 1 for DRUNet, 5 for FastDVDNet.
        repeats_per_sequence: Repetitions of each GT per epoch (clean/synthetic modes).
    """

    def __init__(
        self,
        sequence_info,
        patch_size=None,
        transform=None,
        data_scale=1.0,
        a=1.0,
        b=0.0,
        mode="clean",
        gamma=None,
        num_frames=5,
        repeats_per_sequence=55,
    ):
        self.patch_size = patch_size
        self.transform = transform
        self.data_scale = data_scale
        self.mode = mode
        self.gamma = gamma
        self.num_frames = num_frames
        self.repeats_per_sequence = repeats_per_sequence
        self.stacks = []

        for seq in sequence_info:
            frames = seq["frames"]
            gt = seq["gt"]

            if self.mode == "raw":
                if len(frames) >= self.num_frames:
                    for i in range(len(frames) - (self.num_frames - 1)):
                        self.stacks.append((frames[i : i + self.num_frames], gt))
            elif self.mode in ("synthetic", "clean"):
                if gt:
                    for _ in range(self.repeats_per_sequence):
                        self.stacks.append((None, gt))

    def __len__(self):
        return len(self.stacks)

    def _read_png(self, path):
        img = iio.imread(str(path)).astype(np.float32)
        img = torch.from_numpy(img) / 255.0
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3:
            img = img.permute(2, 0, 1).mean(dim=0, keepdim=True)
        return img  # [1, H, W], range [0, 1]

    def _add_poisson_noise(self, img):
        if self.gamma is None:
            return img
        return torch.poisson(torch.clamp(img, min=0.0) * self.gamma) / self.gamma

    def make_divisible_by_4(self, img):
        H, W = img.shape[-2:]
        return img[..., : (H // 4) * 4, : (W // 4) * 4]

    def _random_crop(self, *tensors):
        """Apply the same random crop to all tensors."""
        H, W = tensors[0].shape[-2:]
        ph, pw = self.patch_size
        if H < ph or W < pw:
            return tensors
        top = torch.randint(0, H - ph + 1, (1,)).item()
        left = torch.randint(0, W - pw + 1, (1,)).item()
        return tuple(t[:, top : top + ph, left : left + pw] for t in tensors)

    def __getitem__(self, idx):
        stack_paths, gt_path = self.stacks[idx]

        if self.mode == "raw":
            frames = [self._read_png(p) for p in stack_paths]
            stack = torch.cat(frames, dim=0)
            target = (
                self._read_png(gt_path)
                if gt_path
                else stack[self.num_frames // 2 : self.num_frames // 2 + 1].clone()
            )
            if self.patch_size is not None:
                stack, target = self._random_crop(stack, target)

        elif self.mode == "clean":
            clean = self._read_png(gt_path)
            if self.patch_size is not None:
                (clean,) = self._random_crop(clean)
            stack = clean.expand(self.num_frames, -1, -1).clone()  # [num_frames, H, W]
            target = clean

        else:  # synthetic
            clean = self._read_png(gt_path)
            if self.patch_size is not None:
                (clean,) = self._random_crop(clean)
            frames = [self._add_poisson_noise(clean) for _ in range(self.num_frames)]
            stack = torch.cat(frames, dim=0)
            target = clean

        stack = self.make_divisible_by_4(stack) / self.data_scale
        stack = torch.clamp(stack, min=0.0)
        target = self.make_divisible_by_4(target) / self.data_scale
        target = torch.clamp(target, min=0.0)

        if self.transform:
            combined = torch.cat([stack, target], dim=0)
            combined = self.transform(combined)
            stack = combined[: self.num_frames]
            target = combined[self.num_frames :]

        return stack, target
