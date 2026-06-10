r"""
Inference-only script: load GR2R/DRUNet weights and denoise sequences listed in a .txt file.

Example:
  python run_inference_on_sequences.py \
    --weights results/denoising-poisson-fmdd-drunet/tif_output_2026_06_01-19_33_02/best_model.pth \
    --dataset fmdd \
    --sequences my_fmdd_sequences.txt \
    --fmdd-root ../data/FMDD

  python run_inference_on_sequences.py \
    --weights results/denoising-poisson-loreal-drunet-from-fmdd/tif_output_.../best_model.pth \
    --dataset loreal \
    --sequences my_loreal_sequences.txt \
    --loreal-data-dir /path/to/sequences_almost_Poisson
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
# from loreal_dataset import linear_transform
from loreal_dataset_fixed import get_fmdd_sequences

from saturation_experiment import add_saturation_patches

BASE_DIR = Path(".")
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "inference"

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


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
    """Return (seq_path, a, b) for one Loreal sequence entry."""
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


def resolve_fmdd_sequence(line: str, fmdd_root: Path) -> dict:
    """Return FMDD sequence dict (same format as get_fmdd_sequences)."""
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

    # Fallback: match against discovered sequences (slower but forgiving)
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
    backbone = dinv.models.DRUNet(
        in_channels=1, out_channels=1, pretrained=None, nc=[16, 32, 64, 128]
    )
    model = dinv.models.ArtifactRemoval(backbone).to(device)
    noise_model = dinv.physics.PoissonNoise(noise)
    noise_model.sigma = noise
    criterion = R2RLoss(noise_model=noise_model, alpha=alpha)
    return criterion.adapt_model(model)


def load_weights(model: torch.nn.Module, weights_path: Path) -> None:
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {k: v for k, v in state_dict.items() if not k.startswith("noise_model.")}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"WARNING: missing keys when loading weights: {len(missing)}")
    if unexpected:
        print(f"WARNING: unexpected keys when loading weights: {len(unexpected)}")



def crop_div16(tensor: torch.Tensor) -> torch.Tensor:
    h, w = tensor.shape[-2:]
    return tensor[..., : (h // 16) * 16, : (w // 16) * 16]


def read_png_tensor(path: Path) -> torch.Tensor:
    img = iio.imread(str(path))
    img = torch.from_numpy(img).float() / 255.0
    if img.ndim == 2:
        img = img.unsqueeze(0)
    elif img.ndim == 3:
        img = img.permute(2, 0, 1).mean(dim=0, keepdim=True)
    return img


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
def run_fmdd_clean(
    model,
    physics,
    seq: dict,
    output_dir: Path,
    save_noisy: bool,
    save_clean: bool,
) -> dict | None:
    """One GT frame + synthetic Poisson noise (matches FMDD training eval)."""
    gt_path = seq.get("gt")
    if not gt_path or not Path(gt_path).exists():
        print(f"  SKIP {seq['modality']}/{seq['seq_id']}: no GT (avg50.png)")
        return None

    x_clean = read_png_tensor(Path(gt_path)).unsqueeze(0).to(device)
    x_clean = crop_div16(x_clean)
    y_noisy = physics(x_clean)
    x_est = model(y_noisy, physics)

    tag = f"{seq['modality']}_{seq['seq_id']}".replace("/", "_")
    tifffile.imwrite(str(output_dir / f"{tag}_denoised.tif"), x_est.squeeze().cpu().numpy().astype(np.float32))
    if save_noisy:
        tifffile.imwrite(str(output_dir / f"{tag}_noisy.tif"), y_noisy.squeeze().cpu().numpy().astype(np.float32))
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
) -> None:
    """Denoise real noisy PNG frames from FMDD raw folder."""
    frames = seq["frames"]
    if not frames:
        print(f"  SKIP {seq['modality']}/{seq['seq_id']}: no raw frames")
        return

    indices = list(range(0, len(frames), frame_stride))
    if max_frames is not None:
        indices = indices[:max_frames]

    denoised_list = []
    noisy_list = []
    for i in tqdm(indices, desc=f"raw {seq['modality']}/{seq['seq_id']}", leave=False):
        y = read_png_tensor(Path(frames[i])).unsqueeze(0).to(device)
        y = crop_div16(y)
        x_est = model(y, physics)
        denoised_list.append(x_est.squeeze().cpu().numpy().astype(np.float32))
        if save_noisy:
            noisy_list.append(y.squeeze().cpu().numpy().astype(np.float32))

    tag = f"{seq['modality']}_{seq['seq_id']}".replace("/", "_")
    stack = np.stack(denoised_list, axis=0)
    tifffile.imwrite(str(output_dir / f"{tag}_denoised.tif"), stack)
    if save_noisy and noisy_list:
        tifffile.imwrite(str(output_dir / f"{tag}_noisy.tif"), np.stack(noisy_list, axis=0))
    print(f"  Saved {tag}_denoised.tif shape={stack.shape}")

@torch.no_grad()
def run_fmdd_patched(model, physics, seq, output_dir, save_noisy, save_clean, patch_seed):
    gt_path = seq.get("gt")
    if not gt_path or not Path(gt_path).exists():
        print(f"  SKIP {seq['modality']}/{seq['seq_id']}: no GT")
        return None

    # Cargar en escala cruda (0-255) para aplicar patches
    gt_np = iio.imread(str(gt_path)).astype(np.float32)
    if gt_np.ndim == 3:
        gt_np = gt_np.mean(axis=-1)

    gt_patched = add_saturation_patches(gt_np, seed=patch_seed)

    # Normalizar y convertir a tensor
    x_clean = torch.from_numpy(gt_patched / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    x_clean = crop_div16(x_clean)

    y_noisy = physics(x_clean)
    y_noisy = torch.clamp(y_noisy, max=1.0)  # saturación

    x_est = model(y_noisy, physics)

    tag = f"{seq['modality']}_{seq['seq_id']}".replace("/", "_")
    tifffile.imwrite(str(output_dir / f"{tag}_denoised.tif"), x_est.squeeze().cpu().numpy().astype(np.float32))
    if save_noisy:
        tifffile.imwrite(str(output_dir / f"{tag}_noisy.tif"), y_noisy.squeeze().cpu().numpy().astype(np.float32))
    if save_clean:
        tifffile.imwrite(str(output_dir / f"{tag}_clean.tif"), x_clean.squeeze().cpu().numpy().astype(np.float32))

    return {
        "sequence": f"{seq['modality']}/{seq['seq_id']}",
        "psnr": PSNR()(x=x_clean, x_net=x_est).item(),
        "ssim": SSIM()(x=x_clean, x_net=x_est).item(),
    }


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
) -> None:
    seq_dir = Path(seq_path)
    tif_files = sorted(seq_dir.glob("*.tif"))
    indices = list(range(0, len(tif_files), frame_stride))
    if max_frames is not None:
        indices = indices[:max_frames]

    denoised_frames = []
    noisy_frames = []
    seq_name = seq_dir.name.replace("/", "_")

    for i in tqdm(indices, desc=seq_name, leave=False):
        img = tifffile.imread(str(tif_files[i])).astype(np.float32)
        y = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
        # y = linear_transform(y, a, b, u=1) / data_scale
        y = y / data_scale
        y = torch.clamp(y, min=0.0)
        y = crop_div16(y)
        x_est = model(y, physics)
        # x_est_orig = linear_transform(x_est * data_scale, a, b, u=1, inverse=True)
        x_est_orig = x_est * data_scale
        denoised_frames.append(x_est_orig.squeeze().cpu().numpy().astype(np.float32))
        if save_noisy:
            noisy_frames.append(y.squeeze().cpu().numpy().astype(np.float32))

    denoised_stack = np.stack(denoised_frames, axis=0)
    tifffile.imwrite(str(output_dir / f"{seq_name}_denoised.tif"), denoised_stack)
    if save_noisy:
        tifffile.imwrite(str(output_dir / f"{seq_name}_noisy.tif"), np.stack(noisy_frames, axis=0))
    print(f"  Saved {seq_name}_denoised.tif shape={denoised_stack.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def default_noise_and_scale(dataset: str) -> tuple[float, float]:
    if dataset == "fmdd":
        return 1 / 255.0, 1.0
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
        description="Run inference with saved GR2R weights on a custom sequence list."
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to .pth state dict")
    parser.add_argument(
        "--dataset",
        choices=["fmdd", "loreal"],
        required=True,
        help="Dataset family for the sequences in the list file",
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        required=True,
        help="Text file with one sequence per line (see module docstring)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder (default: results/inference/<dataset>_<timestamp>)",
    )
    parser.add_argument("--fmdd-root", type=Path, default=Path("../data/FMDD"))
    parser.add_argument(
        "--loreal-data-dir",
        type=Path,
        default=Path("/home/diegosilvera/Escritorio/2026/sequences_almost_Poisson"),
        help="Base dir to resolve Loreal folder names (not full paths)",
    )
    parser.add_argument(
        "--fmdd-mode",
        choices=["clean", "raw", "patched"],
        default="clean",
        help="FMDD: 'clean' = GT+Poisson (training-style); 'raw' = real noisy PNGs",
    )
    parser.add_argument("--noise", type=float, default=None, help="Poisson noise level (default: 1/255)")
    parser.add_argument("--alpha", type=float, default=0.15, help="GR2R alpha used at training")
    parser.add_argument("--data-scale", type=float, default=None)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames per sequence (raw/loreal). Omit for all frames.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every N-th frame (speed up long sequences)",
    )
    parser.add_argument("--no-save-noisy", action="store_true")
    parser.add_argument("--no-save-clean", action="store_true", help="FMDD clean mode only")
    parser.add_argument(
        "--use-params-from-checkpoint-dir",
        action="store_true",
        help="Override noise/alpha/data_scale from parameters.txt next to weights",
    )
    parser.add_argument("--patch-seed", type=int, default=0, help="Para que los patches sean reproducibles",) 
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if args.use_params_from_checkpoint_dir:
        apply_params_from_checkpoint_dir(args)

    default_noise, default_scale = default_noise_and_scale(args.dataset)
    noise = args.noise if args.noise is not None else default_noise
    data_scale = args.data_scale if args.data_scale is not None else default_scale

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = args.output_dir or (DEFAULT_RESULTS_DIR / f"{args.dataset}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = read_sequence_lines(args.sequences)
    print(f"Loaded {len(lines)} sequence(s) from {args.sequences}")
    print(f"Device: {device} | noise={noise} | alpha={args.alpha} | data_scale={data_scale}")

    model = build_model(noise, args.alpha)
    load_weights(model, args.weights)
    model.eval()

    noise_model = dinv.physics.PoissonNoise(noise)
    noise_model.sigma = noise
    physics = dinv.physics.Denoising(noise_model=noise_model)

    save_noisy = not args.no_save_noisy
    save_clean = not args.no_save_clean
    metrics_rows: list[dict] = []

    if args.dataset == "fmdd":
        fmdd_seqs = [resolve_fmdd_sequence(line, args.fmdd_root) for line in lines]
        save_run_config(output_dir, args, [f"{s['modality']}/{s['seq_id']}" for s in fmdd_seqs])

        for seq in fmdd_seqs:
            label = f"{seq['modality']}/{seq['seq_id']}"
            print(f"Processing FMDD {label} ({args.fmdd_mode})...")
            if args.fmdd_mode == "clean":
                row = run_fmdd_clean(model, physics, seq, output_dir, save_noisy, save_clean)
                if row:
                    metrics_rows.append(row)
                    print(f"  PSNR={row['psnr']:.2f} dB  SSIM={row['ssim']:.4f}")
            elif args.fmdd_mode == "patched":
                row = run_fmdd_patched(model, physics, seq, output_dir, save_noisy, save_clean, args.patch_seed)
                if row:
                    metrics_rows.append(row)
                    print(f"  PSNR={row['psnr']:.2f} dB  SSIM={row['ssim']:.4f}")
            else:
                run_fmdd_raw(
                    model,
                    physics,
                    seq,
                    output_dir,
                    args.max_frames,
                    args.frame_stride,
                    save_noisy,
                )
        
    else:
        loreal_seqs = [
            resolve_loreal_sequence(line, args.loreal_data_dir) for line in lines
        ]
        save_run_config(output_dir, args, [s[0] for s in loreal_seqs])

        for seq_path, a, b in loreal_seqs:
            print(f"Processing Loreal {Path(seq_path).name}...")
            run_loreal_sequence(
                model,
                physics,
                seq_path,
                a,
                b,
                output_dir,
                data_scale,
                args.max_frames,
                args.frame_stride,
                save_noisy,
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
