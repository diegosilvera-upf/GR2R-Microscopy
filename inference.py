r"""
Unified GR2R inference: denoise sequences with DRUNet or FastDVDNet weights.

DRUNet processes frames independently (single-frame denoising).
FastDVDNet uses a 5-frame sliding window; central frame is denoised with temporal context.
FastDVDNet also supports geometric TTA (8 D4 transforms) and stochastic R2R ensemble.

Examples:
  # DRUNet on FMDD
  python inference.py --model drunet \
    --weights results/denoising-poisson-fmdd-drunet/.../best_model.pth \
    --dataset fmdd --sequences my_sequences.txt

  # FastDVDNet on Loreal, load noise/alpha from checkpoint folder
  python inference.py --model fastdvdnet \
    --weights results/denoising-poisson-loreal-fastdvdnet/.../best_model.pth \
    --dataset loreal --sequences my_sequences.txt \
    --use-params-from-checkpoint-dir

  # FastDVDNet on Loreal with geometric TTA + stochastic ensemble
  python inference.py --model fastdvdnet \
    --weights results/.../best_model.pth \
    --dataset loreal --sequences my_sequences.txt \
    --geometric-ensemble --n-samples 8

  # DRUNet on generic TIF sequences (must specify data-scale)
  python inference.py --model drunet \
    --weights results/.../best_model.pth \
    --dataset tif-seq --sequences my_sequences.txt --data-scale 255.0
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import deepinv as dinv
import imageio.v3 as iio
import numpy as np
import tifffile
import torch
from deepinv.loss import PSNR, R2RLoss, SSIM
from tqdm import tqdm

from dataset import LorealSequenceDataset, get_fmdd_sequences
from models_FastDVDnet_sans_noise_map import FastDVDnet
from training_utils import FastDVDNetContextWrapper

BASE_DIR = Path(".")
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "inference"

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Geometric TTA helpers (D4 group: 4 rotations × 2 reflections)
# ---------------------------------------------------------------------------


def apply_tta(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    elif mode == 1:
        return torch.rot90(x, 1, [2, 3])
    elif mode == 2:
        return torch.rot90(x, 2, [2, 3])
    elif mode == 3:
        return torch.rot90(x, 3, [2, 3])
    elif mode == 4:
        return torch.flip(x, [3])
    elif mode == 5:
        return torch.flip(torch.rot90(x, 1, [2, 3]), [3])
    elif mode == 6:
        return torch.flip(torch.rot90(x, 2, [2, 3]), [3])
    elif mode == 7:
        return torch.flip(torch.rot90(x, 3, [2, 3]), [3])
    return x


def inv_tta(y: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return y
    elif mode == 1:
        return torch.rot90(y, -1, [2, 3])
    elif mode == 2:
        return torch.rot90(y, -2, [2, 3])
    elif mode == 3:
        return torch.rot90(y, -3, [2, 3])
    elif mode == 4:
        return torch.flip(y, [3])
    elif mode == 5:
        return torch.rot90(torch.flip(y, [3]), -1, [2, 3])
    elif mode == 6:
        return torch.rot90(torch.flip(y, [3]), -2, [2, 3])
    elif mode == 7:
        return torch.rot90(torch.flip(y, [3]), -3, [2, 3])
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


# ---------------------------------------------------------------------------
# Model construction and weight loading
# ---------------------------------------------------------------------------


def build_model(model_type: str, noise: float, alpha: float) -> torch.nn.Module:
    noise_model = dinv.physics.PoissonNoise(noise)
    noise_model.sigma = noise
    criterion = R2RLoss(noise_model=noise_model, alpha=alpha)

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
    # If checkpoint keys don't overlap with model keys, try adding "model." prefix.
    # This handles plain FastDVDNet checkpoints loaded into FastDVDNetContextWrapper.
    model_keys = set(model.state_dict().keys())
    if not (model_keys & set(cleaned.keys())):
        prefixed = {f"model.{k}": v for k, v in cleaned.items()}
        if model_keys & set(prefixed.keys()):
            cleaned = prefixed
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


def save_run_config(output_dir: Path, args: argparse.Namespace, sequences: list) -> None:
    with open(output_dir / "inference_config.txt", "w") as f:
        f.write(f"timestamp={datetime.now().isoformat()}\n")
        f.write(f"device={device}\n")
        for k, v in vars(args).items():
            f.write(f"{k}={v}\n")
        f.write(f"n_sequences={len(sequences)}\n")
        f.write("sequences:\n")
        for s in sequences:
            f.write(f"  {s}\n")


def default_noise_and_scale(dataset: str) -> tuple[float, float | None]:
    if dataset == "fmdd":
        return 1 / 255.0, 1.0
    if dataset == "tif-seq":
        return 1 / 255.0, None  # user must provide --data-scale
    return 1 / 255.0, 255.0  # loreal


# ---------------------------------------------------------------------------
# Blind quality metric: PURE (Poisson Unbiased Risk Estimator)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_pure_mc(
    model: torch.nn.Module,
    y_central: torch.Tensor,
    physics,
    noise_level: float,
    y_stack: torch.Tensor | None = None,
    eps: float | None = None,
    plain: bool = False,
) -> float:
    """MC estimate of per-pixel MSE without a clean reference (PURE for Poisson noise).

    Assumes y ~ Poisson(x / noise_level) * noise_level (deepinv PoissonNoise convention).
    Costs 2 forward passes using a Rademacher perturbation b ∈ {-1, +1}^N:

        PURE = (1/M)[‖f(y)-y‖² + 2·α·(y⊙b)ᵀ(f(y+εb)-f(y))/ε − α·∑ᵢyᵢ]

    where α = noise_level, M = number of pixels, and ε defaults to noise_level
    (= one photon count in the normalized domain).

    The second term estimates the weighted divergence ∑ᵢ yᵢ ∂fᵢ/∂yᵢ — how much
    the denoiser "shrinks" the observation, which corrects the positive bias from
    the first term.

    Args:
        y_central: [1, 1, H, W] central (or only) frame, normalized to [0, 1].
        y_stack:   [1, 5, H, W] full temporal stack for FastDVDNet; None for DRUNet.
                   Only the central frame is perturbed; context frames are fixed.
        eps:       finite-difference step; defaults to noise_level.
        plain:     if True, bypass R2RModel's stochastic test-time ensembling and
                   use a single deterministic pass through the trained backbone
                   (see `plain_forward`) for both forward evaluations.
    """
    if eps is None:
        eps = noise_level

    b = 2 * torch.randint(0, 2, y_central.shape, device=y_central.device, dtype=y_central.dtype) - 1
    y_pert = (y_central + eps * b).clamp(min=0.0)

    if plain:
        f_y = plain_forward(model, y_central, physics, y_stack)
        f_yb = plain_forward(model, y_pert, physics, y_stack)
    elif y_stack is not None:
        # FastDVDNet: set context, perturb only central frame
        model.model.set_context(y_stack)
        f_y = model(y_central, physics)
        model.model.set_context(y_stack)
        f_yb = model(y_pert, physics)
    else:
        f_y = model(y_central, physics)
        f_yb = model(y_pert, physics)

    n_pixels = f_y.numel()
    residual_sq = ((f_y - y_central) ** 2).sum()
    weighted_div = ((y_central * b) * (f_yb - f_y) / eps).sum()
    bias_corr = noise_level * y_central.sum()

    return ((residual_sq + 2 * noise_level * weighted_div - bias_corr) / n_pixels).item()


# ---------------------------------------------------------------------------
# Inference functions (one per dataset type)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_fmdd_clean(
    model,
    model_type: str,
    physics,
    seq: dict,
    output_dir: Path,
    save_noisy: bool,
    save_clean: bool,
    n_samples: int = 1,
    geometric: bool = False,
    plain: bool = False,
) -> dict | None:
    """GT frame + synthetic Poisson noise. Computes PSNR/SSIM against GT."""
    gt_path = seq.get("gt")
    if not gt_path or not Path(gt_path).exists():
        print(f"  SKIP {seq['modality']}/{seq['seq_id']}: no GT (avg50.png)")
        return None

    x_clean = read_png_tensor(Path(gt_path)).unsqueeze(0).to(device)
    x_clean = crop_to_model(x_clean, model_type)

    if model_type == "drunet":
        y_noisy = physics(x_clean)
        x_est = plain_forward(model, y_noisy, physics) if plain else model(y_noisy, physics)
        y_for_save = y_noisy
    else:
        noisy_frames = [physics(x_clean) for _ in range(5)]
        stack_noisy = torch.cat(noisy_frames, dim=1)
        x_est = ensemble_forward(model, stack_noisy, physics, n_samples, geometric, plain=plain)
        y_for_save = stack_noisy[:, 2:3, :, :]

    tag = f"{seq['modality']}_{seq['seq_id']}".replace("/", "_")
    tifffile.imwrite(
        str(output_dir / f"{tag}_denoised.tif"),
        x_est.squeeze().cpu().numpy().astype(np.float32),
    )
    if save_noisy:
        tifffile.imwrite(
            str(output_dir / f"{tag}_noisy.tif"),
            y_for_save.squeeze().cpu().numpy().astype(np.float32),
        )
    if save_clean:
        tifffile.imwrite(
            str(output_dir / f"{tag}_clean.tif"),
            x_clean.squeeze().cpu().numpy().astype(np.float32),
        )

    return {
        "sequence": f"{seq['modality']}/{seq['seq_id']}",
        "psnr": PSNR()(x=x_clean, x_net=x_est).item(),
        "ssim": SSIM()(x=x_clean, x_net=x_est).item(),
    }


@torch.no_grad()
def run_fmdd_raw(
    model,
    model_type: str,
    physics,
    seq: dict,
    output_dir: Path,
    max_frames: int | None,
    frame_stride: int,
    save_noisy: bool,
    n_samples: int = 1,
    geometric: bool = False,
    plain: bool = False,
) -> None:
    """Real noisy PNGs from FMDD raw folder.

    DRUNet: each frame independently.
    FastDVDNet: 5-frame sliding window, denoises central frame.
    """
    frames = seq["frames"]
    tag = f"{seq['modality']}_{seq['seq_id']}".replace("/", "_")
    denoised_list: list[np.ndarray] = []
    noisy_list: list[np.ndarray] = []

    if model_type == "drunet":
        indices = list(range(0, len(frames), frame_stride))
        if max_frames is not None:
            indices = indices[:max_frames]
        for i in tqdm(indices, desc=f"raw {tag}", leave=False):
            y = read_png_tensor(Path(frames[i])).unsqueeze(0).to(device)
            y = crop_to_model(y, model_type)
            x_est = plain_forward(model, y, physics) if plain else model(y, physics)
            denoised_list.append(x_est.squeeze().cpu().numpy().astype(np.float32))
            if save_noisy:
                noisy_list.append(y.squeeze().cpu().numpy().astype(np.float32))

    else:  # fastdvdnet — 5-frame sliding window
        mid = 2
        if len(frames) < 5:
            print(f"  SKIP {tag}: fewer than 5 frames for FastDVDNet")
            return
        center_indices = list(range(mid, len(frames) - mid, frame_stride))
        if max_frames is not None:
            center_indices = center_indices[:max_frames]
        for i in tqdm(center_indices, desc=f"raw {tag}", leave=False):
            window = [
                read_png_tensor(Path(frames[j])).unsqueeze(0).to(device)
                for j in range(i - mid, i + mid + 1)
            ]
            stack = torch.cat(window, dim=1)
            stack = crop_to_model(stack, model_type)
            x_est = ensemble_forward(model, stack, physics, n_samples, geometric, plain=plain)
            denoised_list.append(x_est.squeeze().cpu().numpy().astype(np.float32))
            if save_noisy:
                noisy_list.append(stack[:, 2:3].squeeze().cpu().numpy().astype(np.float32))

    stack_out = np.stack(denoised_list, axis=0)
    tifffile.imwrite(str(output_dir / f"{tag}_denoised.tif"), stack_out)
    if save_noisy and noisy_list:
        tifffile.imwrite(str(output_dir / f"{tag}_noisy.tif"), np.stack(noisy_list, axis=0))
    print(f"  Saved {tag}_denoised.tif shape={stack_out.shape}")


@torch.no_grad()
def run_loreal_sequence(
    model,
    model_type: str,
    physics,
    seq_path: str,
    a: float,
    b: float,
    output_dir: Path,
    data_scale: float,
    max_frames: int | None,
    frame_stride: int,
    save_noisy: bool,
    n_samples: int = 1,
    geometric: bool = False,
    plain: bool = False,
) -> None:
    """Real noisy Loreal TIF frames.

    DRUNet: each frame independently.
    FastDVDNet: LorealSequenceDataset provides 5-frame sliding windows.
    """
    seq_name = Path(seq_path).name
    denoised_frames: list[np.ndarray] = []
    noisy_frames: list[np.ndarray] = []

    if model_type == "drunet":
        tif_files = sorted(Path(seq_path).glob("*.tif"))
        indices = list(range(0, len(tif_files), frame_stride))
        if max_frames is not None:
            indices = indices[:max_frames]
        for i in tqdm(indices, desc=seq_name, leave=False):
            y = read_tif_tensor(tif_files[i], data_scale).unsqueeze(0).to(device)
            y = torch.clamp(y, min=0.0)
            y = crop_to_model(y, model_type)
            x_est = plain_forward(model, y, physics) if plain else model(y, physics)
            denoised_frames.append((x_est * data_scale).squeeze().cpu().numpy().astype(np.float32))
            if save_noisy:
                noisy_frames.append((y * data_scale).squeeze().cpu().numpy().astype(np.float32))

    else:  # fastdvdnet — 5-frame windows via dataset
        seq_ds = LorealSequenceDataset(
            sequence_info=[(seq_path, a, b)],
            num_frames=5,
            data_scale=data_scale,
        )
        if len(seq_ds) == 0:
            print(f"  SKIP {seq_name}: no valid 5-frame windows")
            return
        indices = list(range(0, len(seq_ds), frame_stride))
        if max_frames is not None:
            indices = indices[:max_frames]
        for i in tqdm(indices, desc=seq_name, leave=False):
            y_stack, y_central = seq_ds[i]
            y_stack = y_stack.unsqueeze(0).to(device)
            y_central = y_central.unsqueeze(0).to(device)
            x_est = ensemble_forward(model, y_stack, physics, n_samples, geometric, plain=plain)
            denoised_frames.append((x_est * data_scale).squeeze().cpu().numpy().astype(np.float32))
            if save_noisy:
                noisy_frames.append((y_central * data_scale).squeeze().cpu().numpy().astype(np.float32))

    denoised_stack = np.stack(denoised_frames, axis=0)
    tifffile.imwrite(str(output_dir / f"{seq_name}_denoised.tif"), denoised_stack)
    if save_noisy:
        tifffile.imwrite(str(output_dir / f"{seq_name}_noisy.tif"), np.stack(noisy_frames, axis=0))
    print(f"  Saved {seq_name}_denoised.tif shape={denoised_stack.shape}")


@torch.no_grad()
def run_tif_sequence(
    model,
    model_type: str,
    physics,
    tif_files: list[Path],
    output_dir: Path,
    data_scale: float,
    max_frames: int | None,
    frame_stride: int,
    save_noisy: bool,
    n_samples: int = 1,
    geometric: bool = False,
    plain: bool = False,
) -> None:
    """Generic TIF sequence (not FMDD or Loreal format).

    DRUNet: each frame independently.
    FastDVDNet: 5-frame sliding window.
    """
    seq_name = tif_files[0].parent.name
    denoised_list: list[np.ndarray] = []
    noisy_list: list[np.ndarray] = []

    if model_type == "drunet":
        indices = list(range(0, len(tif_files), frame_stride))
        if max_frames is not None:
            indices = indices[:max_frames]
        for i in tqdm(indices, desc=seq_name, leave=False):
            y = read_tif_tensor(tif_files[i], data_scale).unsqueeze(0).to(device)
            y = torch.clamp(y, min=0.0)
            y = crop_to_model(y, model_type)
            x_est = plain_forward(model, y, physics) if plain else model(y, physics)
            denoised_list.append((x_est * data_scale).squeeze().cpu().numpy().astype(np.float32))
            if save_noisy:
                noisy_list.append((y * data_scale).squeeze().cpu().numpy().astype(np.float32))

    else:  # fastdvdnet — 5-frame sliding window
        mid = 2
        if len(tif_files) < 5:
            print(f"  SKIP {seq_name}: need at least 5 TIF frames for FastDVDNet, found {len(tif_files)}")
            return
        center_indices = list(range(mid, len(tif_files) - mid, frame_stride))
        if max_frames is not None:
            center_indices = center_indices[:max_frames]
        for i in tqdm(center_indices, desc=seq_name, leave=False):
            window = [
                read_tif_tensor(tif_files[j], data_scale).unsqueeze(0).to(device)
                for j in range(i - mid, i + mid + 1)
            ]
            stack = torch.cat(window, dim=1)
            stack = crop_to_model(stack, model_type)
            x_est = ensemble_forward(model, stack, physics, n_samples, geometric, plain=plain)
            denoised_list.append((x_est * data_scale).squeeze().cpu().numpy().astype(np.float32))
            if save_noisy:
                noisy_list.append((stack[:, 2:3] * data_scale).squeeze().cpu().numpy().astype(np.float32))

    stack_out = np.stack(denoised_list, axis=0)
    tifffile.imwrite(str(output_dir / f"{seq_name}_denoised.tif"), stack_out)
    if save_noisy and noisy_list:
        tifffile.imwrite(str(output_dir / f"{seq_name}_noisy.tif"), np.stack(noisy_list, axis=0))
    print(f"  Saved {seq_name}_denoised.tif shape={stack_out.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified GR2R inference: DRUNet (single-frame) or FastDVDNet (5-frame)."
    )
    parser.add_argument(
        "--model",
        choices=["drunet", "fastdvdnet"],
        required=True,
        help="Architecture to use for denoising",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to .pth state dict")
    parser.add_argument(
        "--dataset",
        choices=["fmdd", "loreal", "tif-seq"],
        required=True,
        help="Dataset format of the input sequences",
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        required=True,
        help="Text file with one sequence per line (see module docstring for formats)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fmdd-root", type=Path, default=Path("../data/FMDD"))
    parser.add_argument(
        "--loreal-data-dir",
        type=Path,
        default=Path("/home/diegosilvera/Escritorio/2026/sequences_almost_Poisson"),
        help="Base directory to resolve Loreal folder names",
    )
    parser.add_argument(
        "--fmdd-mode",
        choices=["clean", "raw"],
        default="clean",
        help="FMDD: 'clean' = GT + synthetic Poisson; 'raw' = real noisy PNGs",
    )
    parser.add_argument("--noise", type=float, default=None, help="Poisson noise level (default: 1/255)")
    parser.add_argument("--alpha", type=float, default=0.15, help="GR2R alpha used at training time")
    parser.add_argument("--data-scale", type=float, default=None, help="Pixel scale factor (required for tif-seq)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per sequence (None = all)")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every N-th frame")
    parser.add_argument("--no-save-noisy", action="store_true", help="Skip saving noisy input TIFs")
    parser.add_argument("--no-save-clean", action="store_true", help="Skip saving clean GT TIFs (FMDD clean only)")
    parser.add_argument(
        "--use-params-from-checkpoint-dir",
        action="store_true",
        help="Load noise/alpha/data_scale from parameters.txt next to the weights file",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="FastDVDNet only: stochastic R2R recorruptions to average (1 = disabled)",
    )
    parser.add_argument(
        "--geometric-ensemble",
        action="store_true",
        help="FastDVDNet only: average over all 8 D4 geometric transforms",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Bypass R2RModel's stochastic test-time ensembling (which otherwise averages "
             "eval_n_samples random recorruptions on every forward call). Single deterministic "
             "pass through the trained backbone. Ignores --n-samples.",
    )
    args = parser.parse_args()

    if args.model == "drunet" and (args.n_samples > 1 or args.geometric_ensemble):
        print("WARNING: --n-samples and --geometric-ensemble are only supported with fastdvdnet. Ignoring.")
        args.n_samples = 1
        args.geometric_ensemble = False

    if args.plain and args.n_samples > 1:
        print("WARNING: --plain bypasses R2RModel's stochastic ensembling; --n-samples is ignored.")

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if args.use_params_from_checkpoint_dir:
        apply_params_from_checkpoint_dir(args)

    default_noise, default_scale = default_noise_and_scale(args.dataset)
    noise = args.noise if args.noise is not None else default_noise
    data_scale = args.data_scale if args.data_scale is not None else default_scale
    if args.dataset == "tif-seq" and data_scale is None:
        raise ValueError("--data-scale is required for --dataset tif-seq (e.g. 255.0 or 65535.0)")

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = args.output_dir or (
        DEFAULT_RESULTS_DIR / f"{args.model}_{args.dataset}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = read_sequence_lines(args.sequences)
    n_geom = 8 if args.geometric_ensemble else 1
    print(
        f"Loaded {len(lines)} sequence(s) | model={args.model} | device={device} | "
        f"noise={noise:.5f} | alpha={args.alpha} | data_scale={data_scale}"
    )
    if args.plain:
        print("Mode: plain (R2RModel ensembling bypassed, deterministic backbone forward)")
    elif args.model == "fastdvdnet":
        print(
            f"Ensemble: n_samples={args.n_samples} | "
            f"geometric={'yes' if args.geometric_ensemble else 'no'} | "
            f"total_passes_per_frame={args.n_samples * n_geom}"
        )

    model = build_model(args.model, noise, args.alpha)
    load_weights(model, args.weights)
    model.eval()

    noise_model = dinv.physics.PoissonNoise(noise)
    noise_model.sigma = noise
    physics = dinv.physics.Denoising(noise_model=noise_model)

    save_noisy = not args.no_save_noisy
    save_clean = not args.no_save_clean
    metrics_rows: list[dict] = []

    if args.dataset == "tif-seq":
        tif_seqs = [resolve_tif_sequence(line) for line in lines]
        save_run_config(output_dir, args, [str(s[0].parent) for s in tif_seqs])
        for tif_files in tif_seqs:
            print(f"Processing tif-seq {tif_files[0].parent.name} ({len(tif_files)} frames)...")
            run_tif_sequence(
                model, args.model, physics, tif_files, output_dir,
                data_scale, args.max_frames, args.frame_stride, save_noisy,
                args.n_samples, args.geometric_ensemble, args.plain,
            )

    elif args.dataset == "fmdd":
        fmdd_seqs = [resolve_fmdd_sequence(line, args.fmdd_root) for line in lines]
        save_run_config(output_dir, args, [f"{s['modality']}/{s['seq_id']}" for s in fmdd_seqs])
        for seq in fmdd_seqs:
            label = f"{seq['modality']}/{seq['seq_id']}"
            print(f"Processing FMDD {label} ({args.fmdd_mode})...")
            if args.fmdd_mode == "clean":
                row = run_fmdd_clean(
                    model, args.model, physics, seq, output_dir, save_noisy, save_clean,
                    args.n_samples, args.geometric_ensemble, args.plain,
                )
                if row:
                    metrics_rows.append(row)
                    print(f"  PSNR={row['psnr']:.2f} dB  SSIM={row['ssim']:.4f}")
            else:
                run_fmdd_raw(
                    model, args.model, physics, seq, output_dir,
                    args.max_frames, args.frame_stride, save_noisy,
                    args.n_samples, args.geometric_ensemble, args.plain,
                )

    else:  # loreal
        loreal_seqs = [resolve_loreal_sequence(line, args.loreal_data_dir) for line in lines]
        save_run_config(output_dir, args, [s[0] for s in loreal_seqs])
        for seq_path, a, b in loreal_seqs:
            print(f"Processing Loreal {Path(seq_path).name}...")
            run_loreal_sequence(
                model, args.model, physics, seq_path, a, b, output_dir,
                data_scale, args.max_frames, args.frame_stride, save_noisy,
                args.n_samples, args.geometric_ensemble, args.plain,
            )

    if metrics_rows:
        metrics_path = output_dir / "metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("sequence\tpsnr\tssim\n")
            for row in metrics_rows:
                f.write(f"{row['sequence']}\t{row['psnr']:.4f}\t{row['ssim']:.4f}\n")
        mean_psnr = sum(r["psnr"] for r in metrics_rows) / len(metrics_rows)
        print(f"Mean PSNR: {mean_psnr:.2f} dB — details in {metrics_path}")

    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
