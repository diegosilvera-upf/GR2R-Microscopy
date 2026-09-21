"""
Shared utilities for GR2R inference scripts (inference.py, compute_pure.py).

Building blocks that more than one script needs, or that can be tested on their
own without going through a command line: the forward passes (plain / TTA /
stochastic ensemble), model construction and weight loading, image readers,
and the parsing of sequence lists. The command-line workflow (argument parsing,
choosing defaults, looping over sequences, writing outputs) lives in inference.py.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

import deepinv as dinv
import imageio.v3 as iio
import numpy as np
import tifffile
import torch
from deepinv.loss import R2RLoss

from dataset import LorealSequenceDataset, get_fmdd_sequences
from models_FastDVDnet_sans_noise_map import FastDVDnet
from training_utils import FastDVDNetContextWrapper

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Geometric Test Time Augmentation (TTA) helpers (D4 group: 4 rotations × 2 reflections)
# ---------------------------------------------------------------------------


N_TTA_MODES = 8


def _split_mode(mode: int) -> tuple[int, bool]:
    """TTA mode 0-7 -> (quarter turns, mirrored). Modes 0-3 rotate, modes 4-7 rotate and mirror."""
    if not 0 <= mode < N_TTA_MODES:
        raise ValueError(f"TTA mode must be in 0..{N_TTA_MODES - 1}, got {mode}")
    return mode % 4, mode >= 4


def apply_tta(x: torch.Tensor, mode: int) -> torch.Tensor:
    """Transform a [B, C, H, W] tensor: `mode % 4` quarter-turns, then (if mode >= 4) mirror the width."""
    turns, mirrored = _split_mode(mode)
    if turns:
        x = torch.rot90(x, turns, [2, 3])
    if mirrored:
        x = torch.flip(x, [3])
    return x


def inv_tta(y: torch.Tensor, mode: int) -> torch.Tensor:
    """Undo apply_tta: the same steps in reverse order (un-mirror first, then un-rotate)."""
    turns, mirrored = _split_mode(mode)
    if mirrored:
        y = torch.flip(y, [3])
    if turns:
        y = torch.rot90(y, -turns, [2, 3])
    return y


# ---------------------------------------------------------------------------
# FastDVDNet ensemble forward (geometric TTA + stochastic R2R)
# ---------------------------------------------------------------------------


def plain_forward(
    model: torch.nn.Module,
    y: torch.Tensor,
    physics,
    y_stack: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single deterministic pass through the trained backbone (model.model).

    Bypasses R2RModel's test-time ensembling, which otherwise averages
    `eval_n_samples` random recorruptions of the input on every call to
    `model(...)` — even when denoising a single, already-real measurement.
    """
    backbone = model.model
    if y_stack is not None:
        backbone.set_context(y_stack)
    return backbone(y, physics)


def ensemble_forward(
    model: torch.nn.Module,
    y_stack: torch.Tensor,
    physics,
    n_samples: int,
    geometric: bool,
    plain: bool = False,
) -> torch.Tensor:
    """Average predictions over geometric transforms and/or stochastic recorruptions.

    Setting model.training=True tells the R2RModel adapter to apply a fresh binomial
    recorruption on each call, which is needed for the stochastic ensemble. Only the
    top-level training flag is toggled — the inner FastDVDNet stays in eval mode.

    plain=True bypasses R2RModel entirely (see `plain_forward`): predictions are
    averaged only over geometric transforms (if any), with no stochastic recorruption.
    """
    n_geom = 8 if geometric else 1
    out_sum = torch.zeros(1, 1, y_stack.shape[-2], y_stack.shape[-1], device=device)

    if plain:
        for m in range(n_geom):
            t_stack = apply_tta(y_stack, m)
            t_central = t_stack[:, 2:3, :, :]
            x_est = plain_forward(model, t_central, physics, t_stack)
            out_sum += inv_tta(x_est, m)
        return out_sum / n_geom

    for _ in range(n_samples):
        for m in range(n_geom):
            t_stack = apply_tta(y_stack, m)
            t_central = t_stack[:, 2:3, :, :]
            model.model.set_context(t_stack)

            if n_samples > 1:
                model.training = True
                x_est = model(t_central, physics, update_parameters=True)
                model.training = False
            else:
                x_est = model(t_central, physics)

            out_sum += inv_tta(x_est, m)

    return out_sum / (n_samples * n_geom)


@dataclass(frozen=True)
class Ensemble:
    """How each frame is denoised (the three flags that used to travel as loose arguments).

    n_samples:  stochastic R2R recorruptions to average (FastDVDNet only)
    geometric:  also average over the 8 D4 flips/rotations (FastDVDNet only)
    plain:      skip R2RModel entirely: one deterministic pass through the trained backbone
    """

    n_samples: int = 1
    geometric: bool = False
    plain: bool = False


def denoise(model: torch.nn.Module, model_type: str, x: torch.Tensor, physics, ens: Ensemble) -> torch.Tensor:
    """Denoise one item from a FrameSource: a [1,1,H,W] frame (DRUNet) or a [1,5,H,W] stack (FastDVDNet)."""
    if model_type == "drunet":
        return plain_forward(model, x, physics) if ens.plain else model(x, physics)
    return ensemble_forward(model, x, physics, ens.n_samples, ens.geometric, plain=ens.plain)


# ---------------------------------------------------------------------------
# Model construction and weight loading
# ---------------------------------------------------------------------------


def build_model(model_type: str, noise: float, alpha: float, eval_n_samples: int = 5) -> torch.nn.Module:
    #mix feelings with this function. R2R hard-coded. I avoid this by calling model.model
    noise_model = dinv.physics.PoissonNoise(noise)
    noise_model.sigma = noise
    criterion = R2RLoss(noise_model=noise_model, alpha=alpha, eval_n_samples=eval_n_samples) #This is what's fucking up the inference when using other weights.

    if model_type == "drunet":
        backbone = dinv.models.DRUNet(
            in_channels=1, out_channels=1, pretrained=None, nc=[16, 32, 64, 128]
        )
        model = dinv.models.ArtifactRemoval(backbone).to(device)
    else:  # fastdvdnet
        base = FastDVDnet(num_input_frames=5)
        model = FastDVDNetContextWrapper(base).to(device)

    return criterion.adapt_model(model)


def load_weights(model: torch.nn.Module, weights_path: Path) -> None:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    cleaned = {k: v for k, v in state_dict.items() if not k.startswith("noise_model.")}
    # If checkpoint keys don't overlap with model keys, try adding "model." or
    # "model.model." prefix. This handles plain FastDVDNet checkpoints loaded into
    # FastDVDNetContextWrapper (one level) or further wrapped by R2RModel (two levels).
    model_keys = set(model.state_dict().keys())
    if not (model_keys & set(cleaned.keys())):
        for prefix in ("model.", "model.model."):
            prefixed = {f"{prefix}{k}": v for k, v in cleaned.items()}
            if model_keys & set(prefixed.keys()):
                cleaned = prefixed
                break
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys when loading weights")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys when loading weights")


# ---------------------------------------------------------------------------
# Image I/O and spatial helpers
# ---------------------------------------------------------------------------


def crop_to_model(tensor: torch.Tensor, model_type: str) -> torch.Tensor:
    """Crop spatial dims to a multiple of 16 (DRUNet) or 4 (FastDVDNet)."""
    div = 16 if model_type == "drunet" else 4
    h, w = tensor.shape[-2:]
    return tensor[..., : (h // div) * div, : (w // div) * div]


def read_png_tensor(path: Path) -> torch.Tensor:
    img = iio.imread(str(path)).astype(np.float32)
    t = torch.from_numpy(img) / 255.0
    if t.ndim == 2:
        t = t.unsqueeze(0)
    elif t.ndim == 3:
        t = t.permute(2, 0, 1).mean(dim=0, keepdim=True)
    return t  # [1, H, W]


def read_tif_tensor(path: Path, data_scale: float) -> torch.Tensor:
    img = tifffile.imread(str(path)).astype(np.float32) / data_scale
    t = torch.from_numpy(img)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    elif t.ndim == 3:
        t = t[0:1]
    return t  # [1, H, W]


# ---------------------------------------------------------------------------
# Sequence list parsing
# ---------------------------------------------------------------------------


def read_sequence_lines(sequences_file: Path) -> list[str]:
    lines = []
    with open(sequences_file, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    if not lines:
        raise ValueError(f"No sequences found in {sequences_file}")
    return lines


def _load_preprocessing(path: Path) -> tuple[float, float]:
    params = np.loadtxt(path)
    if params.ndim == 0:
        return float(params), 0.0
    flat = params.flatten()
    return float(flat[0]), float(flat[1])


def resolve_loreal_sequence(line: str, loreal_data_dir: Path | None) -> tuple[str, float, float]:
    candidate = Path(line)
    if not candidate.is_absolute() and loreal_data_dir is not None:
        by_name = loreal_data_dir / line
        if by_name.is_dir():
            candidate = by_name
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Loreal sequence not found: {line!r}. "
            "Use a full path or a folder name with --loreal-data-dir."
        )
    preproc = candidate / "pre-processing.txt"
    if not preproc.exists():
        raise FileNotFoundError(f"Missing pre-processing.txt in {candidate}")
    a, b = _load_preprocessing(preproc)
    if not list(candidate.glob("*.tif")):
        raise FileNotFoundError(f"No .tif frames in {candidate}")
    return str(candidate.resolve()), a, b


def _build_fmdd_dict(root: Path, modality: str, seq_id: str) -> dict:
    raw_dir = root / modality / "raw" / seq_id
    gt_path = root / modality / "gt" / seq_id / "avg50.png"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"FMDD raw folder not found: {raw_dir}")
    png_files = sorted(raw_dir.glob("*.png"))
    return {
        "modality": modality,
        "seq_id": seq_id,
        "frames": [str(p) for p in png_files],
        "gt": str(gt_path) if gt_path.exists() else None,
    }


def resolve_fmdd_sequence(line: str, fmdd_root: Path) -> dict:
    p = Path(line)
    if p.is_dir():
        parts = p.parts
        for marker in ("raw", "gt"):
            if marker in parts:
                idx = parts.index(marker)
                modality = parts[idx - 1]
                seq_id = p.name
                root = Path(*parts[: idx - 1])
                return _build_fmdd_dict(root, modality, seq_id)

    if "/" in line and not p.exists():
        modality, seq_id = line.split("/", 1)
        return _build_fmdd_dict(fmdd_root, modality.strip(), seq_id.strip())

    if p.exists() and p.is_file():
        raise ValueError(f"Expected a directory for FMDD sequence, got file: {p}")

    key = line.strip()
    for seq in get_fmdd_sequences(fmdd_root):
        if f"{seq['modality']}/{seq['seq_id']}" == key:
            return seq
    raise FileNotFoundError(
        f"Could not resolve FMDD sequence {line!r}. "
        "Use Modality/seq_id, a path to .../raw/N, or a full raw directory."
    )


def resolve_tif_sequence(line: str) -> list[Path]:
    p = Path(line)
    if not p.is_dir():
        raise FileNotFoundError(f"tif-seq directory not found: {line!r}")
    tifs = sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))
    if not tifs:
        raise ValueError(f"{p}: no TIF files found")
    return tifs


# ---------------------------------------------------------------------------
# Frame sources: what the network is fed, one sequence at a time
# ---------------------------------------------------------------------------

N_FRAMES = 5  # temporal window of FastDVDNet


class SkipSequence(Exception):
    """A sequence that cannot be processed (e.g. too few frames). The message says why."""


class FrameSource:
    """The frames of ONE sequence, ready for the network.

    Iterating yields `(x, central)` tensors already on `device`:
      x        network input: a [1,1,H,W] frame for DRUNet, the [1,5,H,W] temporal stack for FastDVDNet
      central  [1,1,H,W] the frame being denoised (this is what gets saved as the "noisy" output)
    Frames are read lazily, one item at a time. `name` is used for the output file names.
    """

    def __init__(self, name: str, indices: list[int], load: Callable[[int], tuple[torch.Tensor, torch.Tensor]]):
        self.name = name
        self.indices = indices
        self._load = load

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for i in self.indices:
            yield self._load(i)


def sliding_indices(start: int, stop: int, stride: int, max_frames: int | None) -> list[int]:
    """range(start, stop, stride), keeping at most `max_frames` entries (None = all)."""
    indices = list(range(start, stop, stride))
    return indices if max_frames is None else indices[:max_frames]


def frame_file_source(
    name: str,
    paths: list[Path],
    read: Callable[[Path], torch.Tensor],
    model_type: str,
    *,
    stride: int,
    max_frames: int | None,
    clamp: bool,
) -> FrameSource:
    """Frames stored one file per frame (PNG or TIF), in temporal order.

    DRUNet denoises every `stride`-th frame on its own. FastDVDNet needs a 5-frame window,
    so only frames 2 .. N-3 can be the centre, and the input is the window [i-2 .. i+2].
    `read` turns a path into a [1,H,W] tensor already scaled to [0,1].
    `clamp` sets negative values to 0 (a TIF can go below 0 after preprocessing).
    """

    def prepare(t: torch.Tensor) -> torch.Tensor:
        if clamp:
            t = torch.clamp(t, min=0.0)
        return crop_to_model(t, model_type)

    if model_type == "drunet":
        indices = sliding_indices(0, len(paths), stride, max_frames)

        def load(i: int):
            y = prepare(read(paths[i]).unsqueeze(0).to(device))
            return y, y

    else:
        if len(paths) < N_FRAMES:
            raise SkipSequence(f"{name}: {len(paths)} frame(s), FastDVDNet needs at least {N_FRAMES}")
        half = N_FRAMES // 2
        indices = sliding_indices(half, len(paths) - half, stride, max_frames)

        def load(i: int):
            window = [read(paths[j]).unsqueeze(0).to(device) for j in range(i - half, i + half + 1)]
            stack = prepare(torch.cat(window, dim=1))
            return stack, stack[:, half : half + 1]

    return FrameSource(name, indices, load)


def loreal_window_source(
    name: str, seq_path: str, a: float, b: float, *, data_scale: float, stride: int, max_frames: int | None
) -> FrameSource:
    """FastDVDNet windows of a Loreal sequence, as built by LorealSequenceDataset.

    The dataset also splits two-channel sequences (`_c0_` / `_c1_` in the file names) into
    separate temporal sequences, and crops each stack to a multiple of 4.
    """
    ds = LorealSequenceDataset(sequence_info=[(seq_path, a, b)], num_frames=N_FRAMES, data_scale=data_scale)
    if len(ds) == 0:
        raise SkipSequence(f"{name}: no valid {N_FRAMES}-frame windows")
    indices = sliding_indices(0, len(ds), stride, max_frames)
    half = N_FRAMES // 2

    def load(i: int):
        stack, _ = ds[i]
        stack = stack.unsqueeze(0).to(device)
        return stack, stack[:, half : half + 1]

    return FrameSource(name, indices, load)


def build_source(
    dataset: str, seq, model_type: str, *, data_scale: float, stride: int, max_frames: int | None
) -> FrameSource:
    """Turn what resolve_*_sequence returned into a FrameSource.

    All the differences between the datasets live here:
      fmdd     PNG frames, already in [0,1] (so `data_scale` does not apply), never clamped
      loreal   DRUNet: sorted *.tif, clamped.  FastDVDNet: windows from LorealSequenceDataset
      tif-seq  TIF frames; clamped for DRUNet but NOT for FastDVDNet. compute_pure.py does clamp
               FastDVDNet windows, so the same data is treated differently by the two scripts.
    """
    opts = dict(stride=stride, max_frames=max_frames)
    read_tif = partial(read_tif_tensor, data_scale=data_scale)
    if dataset == "fmdd":
        name = f"{seq['modality']}_{seq['seq_id']}".replace("/", "_")
        return frame_file_source(name, seq["frames"], read_png_tensor, model_type, clamp=False, **opts)
    if dataset == "loreal":
        seq_path, a, b = seq
        name = Path(seq_path).name
        if model_type == "drunet":
            return frame_file_source(name, sorted(Path(seq_path).glob("*.tif")), read_tif, model_type, clamp=True, **opts)
        return loreal_window_source(name, seq_path, a, b, data_scale=data_scale, **opts)
    if dataset == "tif-seq":
        return frame_file_source(seq[0].parent.name, seq, read_tif, model_type, clamp=(model_type == "drunet"), **opts)
    raise ValueError(f"Unknown dataset: {dataset!r}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def parse_params_file(params_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not params_path.exists():
        return out
    with open(params_path, "r") as f:
        for raw in f:
            m = re.match(r"^(\w+)\s*=\s*(.+)$", raw.strip())
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


def apply_params_from_checkpoint_dir(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.weights).resolve().parent
    params = parse_params_file(ckpt_dir / "parameters.txt")
    if not params:
        return
    print(f"Reading defaults from {ckpt_dir / 'parameters.txt'}")
    if "gamma" in params and args.noise is None:
        args.noise = float(params["gamma"])
    if "noise" in params and args.noise is None:
        args.noise = float(params["noise"])
    if "alpha" in params:
        args.alpha = float(params["alpha"])
    if "data_scale" in params:
        args.data_scale = float(params["data_scale"])
