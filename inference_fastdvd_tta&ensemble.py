r"""
Inference script with Geometric TTA and Stochastic R2R Ensemble for FastDVDNet.

Based on run_inference_on_sequences_fastdvdnet.py, adds two optional refinements:
  --geometric-ensemble   Average 8 geometric transforms (D4 group) per forward pass
  --n-samples N          Average N stochastic R2R recorruptions (N=1 = no stochastic ensemble)

Both can be combined. A single pass with neither flag matches the baseline script exactly.
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

# from loreal_dataset import LorealSequenceDataset
# from loreal_dataset_fixed import get_fmdd_sequences
from dataset import LorealSequenceDataset, get_fmdd_sequences
from models_FastDVDnet_sans_noise_map import FastDVDnet
from training_utils import FastDVDNetContextWrapper

BASE_DIR = Path(".")
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "inference_fastdvdnet"

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Geometric TTA helpers (ported from deprecated test4.py — math verified)
# ---------------------------------------------------------------------------


def apply_tta(x: torch.Tensor, mode: int) -> torch.Tensor:
    """Apply one of 8 D4 symmetries to a [B, C, H, W] tensor."""
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
    """Invert the geometric transform applied by apply_tta(mode)."""
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
# Core ensemble forward pass
# ---------------------------------------------------------------------------


def ensemble_forward(
    model: torch.nn.Module,
    y_stack: torch.Tensor,
    physics,
    n_samples: int,
    geometric: bool,
) -> torch.Tensor:
    """
    Run model with optional geometric TTA and/or stochastic R2R ensemble.

    For stochastic ensemble (n_samples > 1): sets model.training=True so deepinv's
    R2RModel adapter applies a fresh binomial recorruption on each call, matching
    exactly the training distribution. Only the top-level flag is set (not recursive),
    so the inner FastDVDNet stays in eval mode.

    For geometric TTA: applies all 8 D4 symmetries to the 5-frame stack and averages
    the inverse-transformed outputs.
    """
    n_geom = 8 if geometric else 1
    out_sum = torch.zeros(1, 1, y_stack.shape[-2], y_stack.shape[-1], device=device)

    for _ in range(n_samples):
        for m in range(n_geom):
            t_stack = apply_tta(y_stack, m)
            t_central = t_stack[:, 2:3, :, :]

            model.model.set_context(t_stack)

            if n_samples > 1:
                # Training flag activates R2R binomial recorruption inside the adapter.
                # Only sets the flag on the top-level module — inner net stays in eval.
                model.training = True
                x_est = model(t_central, physics, update_parameters=True)
                model.training = False
            else:
                x_est = model(t_central, physics)

            out_sum += inv_tta(x_est, m)

    return out_sum / (n_samples * n_geom)


# ---------------------------------------------------------------------------
# Sequence list parsing (identical to baseline script)
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
    tifs = list(candidate.glob("*.tif"))
    if not tifs:
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


def resolve_tif_sequence(line: str) -> list[Path]:
    p = Path(line)
    if not p.is_dir():
        raise FileNotFoundError(f"tif-seq directory not found: {line!r}")
    tifs = sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))
    if len(tifs) < 5:
        raise ValueError(f"{p}: need at least 5 TIF frames, found {len(tifs)}")
    return tifs


def resolve_fmdd_sequence(line: str, fmdd_root: Path) -> dict:
    p = Path(line)
    if p.is_dir():
        parts = p.parts
        if "raw" in parts:
            idx = parts.index("raw")
            modality = parts[idx - 1]
            seq_id = p.name
            root = Path(*parts[: idx - 1])
            return _build_fmdd_dict(root, modality, seq_id)
        if "gt" in parts:
            idx = parts.index("gt")
            modality = parts[idx - 1]
            seq_id = p.name
            root = Path(*parts[: idx - 1])
            return _build_fmdd_dict(root, modality, seq_id)

    if "/" in line and not p.exists():
        modality, seq_id = line.split("/", 1)
        modality, seq_id = modality.strip(), seq_id.strip()
        return _build_fmdd_dict(fmdd_root, modality, seq_id)

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


# ---------------------------------------------------------------------------
# Model / I/O helpers
# ---------------------------------------------------------------------------


def build_model(noise: float, alpha: float) -> torch.nn.Module:
    base = FastDVDnet(num_input_frames=5)
    wrapper = FastDVDNetContextWrapper(base).to(device)
    noise_model = dinv.physics.PoissonNoise(noise)
    noise_model.sigma = noise
    criterion = R2RLoss(noise_model=noise_model, alpha=alpha)
    return criterion.adapt_model(wrapper)


def load_weights(model: torch.nn.Module, weights_path: Path) -> None:
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {k: v for k, v in state_dict.items() if not k.startswith("noise_model.")}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"WARNING: missing keys when loading weights: {len(missing)}")
    if unexpected:
        print(f"WARNING: unexpected keys when loading weights: {len(unexpected)}")


def crop_div4(tensor: torch.Tensor) -> torch.Tensor:
    h, w = tensor.shape[-2:]
    return tensor[..., :(h // 4) * 4, :(w // 4) * 4]


def read_png_tensor(path: Path) -> torch.Tensor:
    img = iio.imread(str(path))
    img = torch.from_numpy(img).float() / 255.0
    if img.ndim == 2:
        img = img.unsqueeze(0)
    elif img.ndim == 3:
        img = img.permute(2, 0, 1).mean(dim=0, keepdim=True)
    return img


def read_tif_tensor(path: Path, data_scale: float) -> torch.Tensor:
    img = tifffile.imread(str(path)).astype(np.float32) / data_scale
    t = torch.from_numpy(img)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    elif t.ndim == 3:
        t = t[0:1]
    return t


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


# ---------------------------------------------------------------------------
# Inference per dataset
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_tif_sequence(
    model,
    physics,
    tif_files: list[Path],
    output_dir: Path,
    data_scale: float,
    max_frames: int | None,
    frame_stride: int,
    save_noisy: bool,
    n_samples: int,
    geometric: bool,
) -> None:
    mid = 2
    center_indices = list(range(mid, len(tif_files) - mid, frame_stride))
    if max_frames is not None:
        center_indices = center_indices[:max_frames]

    denoised_list: list[np.ndarray] = []
    noisy_list: list[np.ndarray] = []
    seq_name = tif_files[0].parent.name

    for i in tqdm(center_indices, desc=seq_name, leave=False):
        window = [
            read_tif_tensor(tif_files[j], data_scale).unsqueeze(0).to(device)
            for j in range(i - mid, i + mid + 1)
        ]
        stack = torch.cat(window, dim=1)
        stack = crop_div4(stack)
        y_central = stack[:, 2:3, :, :]

        x_est = ensemble_forward(model, stack, physics, n_samples, geometric)
        denoised_list.append((x_est * data_scale).squeeze().cpu().numpy().astype(np.float32))
        if save_noisy:
            noisy_list.append((y_central * data_scale).squeeze().cpu().numpy().astype(np.float32))

    stack_out = np.stack(denoised_list, axis=0)
    tifffile.imwrite(str(output_dir / f"{seq_name}_denoised.tif"), stack_out)
    if save_noisy and noisy_list:
        tifffile.imwrite(str(output_dir / f"{seq_name}_noisy.tif"), np.stack(noisy_list, axis=0))
    print(f"  Saved {seq_name}_denoised.tif shape={stack_out.shape}")


@torch.no_grad()
def run_fmdd_clean(
    model,
    physics,
    seq: dict,
    output_dir: Path,
    save_noisy: bool,
    save_clean: bool,
    n_samples: int,
    geometric: bool,
) -> dict | None:
    gt_path = seq.get("gt")
    if not gt_path or not Path(gt_path).exists():
        print(f"  SKIP {seq['modality']}/{seq['seq_id']}: no GT (avg50.png)")
        return None

    x_clean = read_png_tensor(Path(gt_path)).unsqueeze(0).to(device)
    x_clean = crop_div4(x_clean)

    noisy_frames = [physics(x_clean) for _ in range(5)]
    stack_noisy = torch.cat(noisy_frames, dim=1)
    y_central = stack_noisy[:, 2:3, :, :]

    x_est = ensemble_forward(model, stack_noisy, physics, n_samples, geometric)

    tag = f"{seq['modality']}_{seq['seq_id']}".replace("/", "_")
    tifffile.imwrite(str(output_dir / f"{tag}_denoised.tif"), x_est.squeeze().cpu().numpy().astype(np.float32))
    if save_noisy:
        tifffile.imwrite(str(output_dir / f"{tag}_noisy.tif"), y_central.squeeze().cpu().numpy().astype(np.float32))
    if save_clean:
        tifffile.imwrite(str(output_dir / f"{tag}_clean.tif"), x_clean.squeeze().cpu().numpy().astype(np.float32))

    return {
        "sequence": f"{seq['modality']}/{seq['seq_id']}",
        "psnr": PSNR()(x=x_clean, x_net=x_est).item(),
        "ssim": SSIM()(x=x_clean, x_net=x_est).item(),
    }


@torch.no_grad()
def run_fmdd_raw(
    model,
    physics,
    seq: dict,
    output_dir: Path,
    max_frames: int | None,
    frame_stride: int,
    save_noisy: bool,
    n_samples: int,
    geometric: bool,
) -> None:
    frames = seq["frames"]
    mid = 2
    if len(frames) < 5:
        print(f"  SKIP {seq['modality']}/{seq['seq_id']}: fewer than 5 frames")
        return

    center_indices = list(range(mid, len(frames) - mid, frame_stride))
    if max_frames is not None:
        center_indices = center_indices[:max_frames]

    denoised_list = []
    noisy_list = []
    for i in tqdm(center_indices, desc=f"raw {seq['modality']}/{seq['seq_id']}", leave=False):
        window = [read_png_tensor(Path(frames[j])).unsqueeze(0).to(device)
                  for j in range(i - mid, i + mid + 1)]
        stack = torch.cat(window, dim=1)
        stack = crop_div4(stack)
        y_central = stack[:, 2:3, :, :]

        x_est = ensemble_forward(model, stack, physics, n_samples, geometric)
        denoised_list.append(x_est.squeeze().cpu().numpy().astype(np.float32))
        if save_noisy:
            noisy_list.append(y_central.squeeze().cpu().numpy().astype(np.float32))

    tag = f"{seq['modality']}_{seq['seq_id']}".replace("/", "_")
    stack_out = np.stack(denoised_list, axis=0)
    tifffile.imwrite(str(output_dir / f"{tag}_denoised.tif"), stack_out)
    if save_noisy and noisy_list:
        tifffile.imwrite(str(output_dir / f"{tag}_noisy.tif"), np.stack(noisy_list, axis=0))
    print(f"  Saved {tag}_denoised.tif shape={stack_out.shape}")


@torch.no_grad()
def run_loreal_sequence(
    model,
    physics,
    seq_path: str,
    a: float,
    b: float,
    output_dir: Path,
    data_scale: float,
    max_frames: int | None,
    frame_stride: int,
    save_noisy: bool,
    n_samples: int,
    geometric: bool,
) -> None:
    seq_ds = LorealSequenceDataset(
        sequence_info=[(seq_path, a, b)],
        num_frames=5,
        data_scale=data_scale,
    )
    if len(seq_ds) == 0:
        print(f"  SKIP {Path(seq_path).name}: no valid 5-frame windows")
        return

    indices = list(range(0, len(seq_ds), frame_stride))
    if max_frames is not None:
        indices = indices[:max_frames]

    denoised_frames = []
    noisy_frames = []
    seq_name = Path(seq_path).name

    for i in tqdm(indices, desc=seq_name, leave=False):
        y_stack, y_central = seq_ds[i]
        y_stack = y_stack.unsqueeze(0).to(device)
        y_central = y_central.unsqueeze(0).to(device)

        x_est = ensemble_forward(model, y_stack, physics, n_samples, geometric)

        x_est_orig = x_est * data_scale
        denoised_frames.append(x_est_orig.squeeze().cpu().numpy().astype(np.float32))
        if save_noisy:
            noisy_frames.append((y_central * data_scale).squeeze().cpu().numpy().astype(np.float32))

    denoised_stack = np.stack(denoised_frames, axis=0)
    tifffile.imwrite(str(output_dir / f"{seq_name}_denoised.tif"), denoised_stack)
    if save_noisy:
        tifffile.imwrite(str(output_dir / f"{seq_name}_noisy.tif"), np.stack(noisy_frames, axis=0))
    print(f"  Saved {seq_name}_denoised.tif shape={denoised_stack.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def default_noise_and_scale(dataset: str) -> tuple[float, float | None]:
    if dataset == "fmdd":
        return 1 / 255.0, 1.0
    if dataset == "tif-seq":
        return 1 / 255.0, None  # data_scale must be set explicitly by the user
    return 1 / 255.0, 255.0


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FastDVDNet inference with geometric TTA and/or stochastic R2R ensemble."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", choices=["fmdd", "loreal", "tif-seq"], required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fmdd-root", type=Path, default=Path("../data/FMDD"))
    parser.add_argument(
        "--loreal-data-dir",
        type=Path,
        default=Path("/home/diegosilvera/Escritorio/2026/sequences_almost_Poisson"),
    )
    parser.add_argument("--fmdd-mode", choices=["clean", "raw"], default="clean")
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--data-scale", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-save-noisy", action="store_true")
    parser.add_argument("--no-save-clean", action="store_true")
    parser.add_argument("--use-params-from-checkpoint-dir", action="store_true")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of stochastic R2R recorruptions to average. 1 = no stochastic ensemble.",
    )
    parser.add_argument(
        "--geometric-ensemble",
        action="store_true",
        help="Average over 8 D4 geometric transforms (rotations + flips).",
    )
    args = parser.parse_args()

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
    output_dir = args.output_dir or (DEFAULT_RESULTS_DIR / f"{args.dataset}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = read_sequence_lines(args.sequences)
    n_geom = 8 if args.geometric_ensemble else 1
    print(
        f"Loaded {len(lines)} sequence(s) | device={device} | noise={noise} | "
        f"alpha={args.alpha} | data_scale={data_scale} | "
        f"n_samples={args.n_samples} | geometric={'yes' if args.geometric_ensemble else 'no'} | "
        f"total_passes_per_frame={args.n_samples * n_geom}"
    )

    model = build_model(noise, args.alpha)
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
                model, physics, tif_files, output_dir,
                data_scale, args.max_frames, args.frame_stride, save_noisy,
                args.n_samples, args.geometric_ensemble,
            )
    elif args.dataset == "fmdd":
        fmdd_seqs = [resolve_fmdd_sequence(line, args.fmdd_root) for line in lines]
        save_run_config(output_dir, args, [f"{s['modality']}/{s['seq_id']}" for s in fmdd_seqs])

        for seq in fmdd_seqs:
            label = f"{seq['modality']}/{seq['seq_id']}"
            print(f"Processing FMDD {label} ({args.fmdd_mode})...")
            if args.fmdd_mode == "clean":
                row = run_fmdd_clean(
                    model, physics, seq, output_dir, save_noisy, save_clean,
                    args.n_samples, args.geometric_ensemble,
                )
                if row:
                    metrics_rows.append(row)
                    print(f"  PSNR={row['psnr']:.2f} dB  SSIM={row['ssim']:.4f}")
            else:
                run_fmdd_raw(
                    model, physics, seq, output_dir,
                    args.max_frames, args.frame_stride, save_noisy,
                    args.n_samples, args.geometric_ensemble,
                )
    else:
        loreal_seqs = [resolve_loreal_sequence(line, args.loreal_data_dir) for line in lines]
        save_run_config(output_dir, args, [s[0] for s in loreal_seqs])

        for seq_path, a, b in loreal_seqs:
            print(f"Processing Loreal {Path(seq_path).name}...")
            run_loreal_sequence(
                model, physics, seq_path, a, b, output_dir,
                data_scale, args.max_frames, args.frame_stride, save_noisy,
                args.n_samples, args.geometric_ensemble,
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
