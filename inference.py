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

Sequence list (--sequences): a text file with one sequence per line; blank lines and
lines starting with '#' are ignored. What a line means depends on --dataset:
  fmdd     Modality/seq_id (looked up under --fmdd-root), or a path to .../<Modality>/raw/<seq_id>
  loreal   a folder name (looked up under --loreal-data-dir) or a full path; the folder must
           contain the .tif frames and a pre-processing.txt
  tif-seq  a path to a folder with the .tif/.tiff frames (all .tif first, then all .tiff, each sorted)

Outputs (in --output-dir, default results/inference/<model>_<dataset>_<timestamp>):
  <name>_denoised.tif   the denoised frames, one page per frame, in the units of the input
  <name>_noisy.tif      the input frames that were denoised (skip with --no-save-noisy)
  inference_config.txt  every argument of the run, for reproducibility
  metrics.txt           PSNR/SSIM per sequence (only --dataset fmdd --fmdd-mode clean)

The building blocks (forward passes, model loading, readers, sequence parsing) live in
inference_utils.py; this file is the command line and the loop over sequences.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import deepinv as dinv
import numpy as np
import tifffile
import torch
from deepinv.loss import PSNR, SSIM
from tqdm import tqdm

from inference_utils import (
    N_FRAMES,
    Ensemble,
    FrameSource,
    SkipSequence,
    apply_params_from_checkpoint_dir,
    build_model,
    build_source,
    crop_to_model,
    denoise,
    device,
    load_weights,
    read_png_tensor,
    read_sequence_lines,
    resolve_fmdd_sequence,
    resolve_loreal_sequence,
    resolve_tif_sequence,
)

DEFAULT_RESULTS_DIR = Path("results") / "inference"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


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


def resolve_noise_and_scale(args: argparse.Namespace) -> tuple[float, float | None]:
    """(noise, data_scale) from the command line, falling back to the dataset's defaults.

    Poisson noise defaults to 1/255 everywhere. data_scale (divisor that maps pixel values to
    [0,1]) defaults to 1 for FMDD (its PNGs are normalised by the reader) and 255 for Loreal;
    for tif-seq the bit depth is unknown, so it must be given (255 for 8-bit, 65535 for 16-bit).
    """
    default_scale = {"fmdd": 1.0, "loreal": 255.0, "tif-seq": None}[args.dataset]
    noise = args.noise if args.noise is not None else 1 / 255.0
    data_scale = args.data_scale if args.data_scale is not None else default_scale
    if args.dataset == "tif-seq" and data_scale is None:
        raise ValueError("--data-scale is required for --dataset tif-seq (e.g. 255.0 or 65535.0)")
    return noise, data_scale


def describe_sequence(dataset: str, seq, fmdd_mode: str) -> tuple[str, str]:
    """(label written to inference_config.txt, message printed before processing the sequence)."""
    if dataset == "fmdd":
        label = f"{seq['modality']}/{seq['seq_id']}"
        return label, f"Processing FMDD {label} ({fmdd_mode})..."
    if dataset == "loreal":
        return seq[0], f"Processing Loreal {Path(seq[0]).name}..."
    return str(seq[0].parent), f"Processing tif-seq {seq[0].parent.name} ({len(seq)} frames)..."


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.squeeze().cpu().numpy().astype(np.float32)


@torch.no_grad()
def run_fmdd_clean(
    model,
    model_type: str,
    physics,
    seq: dict,
    output_dir: Path,
    save_noisy: bool,
    save_clean: bool,
    ens: Ensemble,
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
        x_est = denoise(model, model_type, y_noisy, physics, ens)
        y_for_save = y_noisy
    else:
        noisy_frames = [physics(x_clean) for _ in range(N_FRAMES)]
        stack_noisy = torch.cat(noisy_frames, dim=1)
        x_est = denoise(model, model_type, stack_noisy, physics, ens)
        y_for_save = stack_noisy[:, 2:3, :, :]

    tag = f"{seq['modality']}_{seq['seq_id']}".replace("/", "_")
    tifffile.imwrite(str(output_dir / f"{tag}_denoised.tif"), _to_numpy(x_est))
    if save_noisy:
        tifffile.imwrite(str(output_dir / f"{tag}_noisy.tif"), _to_numpy(y_for_save))
    if save_clean:
        tifffile.imwrite(str(output_dir / f"{tag}_clean.tif"), _to_numpy(x_clean))

    return {
        "sequence": f"{seq['modality']}/{seq['seq_id']}",
        "psnr": PSNR()(x=x_clean, x_net=x_est).item(),
        "ssim": SSIM()(x=x_clean, x_net=x_est).item(),
    }


@torch.no_grad()
def run_sequence(
    model,
    model_type: str,
    physics,
    source: FrameSource,
    output_dir: Path,
    data_scale: float,
    save_noisy: bool,
    ens: Ensemble,
) -> None:
    """Denoise every frame of `source` and save it as one multi-page TIF.

    Outputs are multiplied back by `data_scale`, i.e. returned in the units of the input file.
    """
    denoised: list[np.ndarray] = []
    noisy: list[np.ndarray] = []
    for x, central in tqdm(source, total=len(source), desc=source.name, leave=False):
        denoised.append(_to_numpy(denoise(model, model_type, x, physics, ens) * data_scale))
        if save_noisy:
            noisy.append(_to_numpy(central * data_scale))

    stack_out = np.stack(denoised, axis=0)
    tifffile.imwrite(str(output_dir / f"{source.name}_denoised.tif"), stack_out)
    if save_noisy:
        tifffile.imwrite(str(output_dir / f"{source.name}_noisy.tif"), np.stack(noisy, axis=0))
    print(f"  Saved {source.name}_denoised.tif shape={stack_out.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_metrics(output_dir: Path, rows: list[dict]) -> None:
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("sequence\tpsnr\tssim\n")
        for row in rows:
            f.write(f"{row['sequence']}\t{row['psnr']:.4f}\t{row['ssim']:.4f}\n")
    mean_psnr = sum(r["psnr"] for r in rows) / len(rows)
    print(f"Mean PSNR: {mean_psnr:.2f} dB — details in {metrics_path}")


def build_parser() -> argparse.ArgumentParser:
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
        help="Text file with one sequence per line ('#' comments allowed). fmdd: Modality/seq_id or "
             ".../raw/<seq_id>; loreal: folder name (looked up in --loreal-data-dir) or full path; "
             "tif-seq: path to a folder of .tif/.tiff frames",
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
        help="FastDVDNet only. 1 (default): one call to R2RModel, which itself averages its eval_n_samples "
             "(5) random recorruptions. N>1: N calls with one recorruption each, averaged. "
             "Use --plain for a fully deterministic pass.",
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
    return parser


def main() -> None:
    args = build_parser().parse_args()

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

    noise, data_scale = resolve_noise_and_scale(args)

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

    ens = Ensemble(args.n_samples, args.geometric_ensemble, args.plain)
    save_noisy = not args.no_save_noisy
    save_clean = not args.no_save_clean
    metrics_rows: list[dict] = []

    resolve = {
        "fmdd": lambda line: resolve_fmdd_sequence(line, args.fmdd_root),
        "loreal": lambda line: resolve_loreal_sequence(line, args.loreal_data_dir),
        "tif-seq": resolve_tif_sequence,
    }[args.dataset]
    sequences = [resolve(line) for line in lines]
    described = [describe_sequence(args.dataset, seq, args.fmdd_mode) for seq in sequences]
    save_run_config(output_dir, args, [label for label, _ in described])

    # FMDD PNGs are already normalised to [0,1] by the reader: --data-scale only applies to TIF data.
    out_scale = 1.0 if args.dataset == "fmdd" else data_scale

    for seq, (_, message) in zip(sequences, described):
        print(message)
        if args.dataset == "fmdd" and args.fmdd_mode == "clean":
            row = run_fmdd_clean(model, args.model, physics, seq, output_dir, save_noisy, save_clean, ens)
            if row:
                metrics_rows.append(row)
                print(f"  PSNR={row['psnr']:.2f} dB  SSIM={row['ssim']:.4f}")
            continue
        try:
            source = build_source(
                args.dataset, seq, args.model,
                data_scale=out_scale, stride=args.frame_stride, max_frames=args.max_frames,
            )
        except SkipSequence as e:
            print(f"  SKIP {e}")
            continue
        run_sequence(model, args.model, physics, source, output_dir, out_scale, save_noisy, ens)

    if metrics_rows:
        write_metrics(output_dir, metrics_rows)

    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
