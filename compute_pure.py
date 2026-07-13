"""
Standalone PURE (Poisson Unbiased Risk Estimator) script.


Estimates per-pixel MSE without clean ground truth. Assumes pure Poisson noise
y ~ Poisson(x) in the photon count domain (γ=1), with data normalized to [0,1]
by dividing by data_scale. The effective noise parameter is noise_level = 1/data_scale.


Each frame costs K × 2 forward passes (K = --n-b). At inference without gradients
this is cheap, so K=8 or K=16 is a reasonable default. Variance scales as 1/(M·K)
where M is the number of pixels; returns diminish fast for large images.


For FMDD --fmdd-mode clean: also computes true MSE alongside PURE, which lets
you validate that the estimator is working correctly (PURE ≈ true MSE in expectation).


Usage:
 # Loreal sequences (blind, no GT)
 python compute_pure.py --model fastdvdnet \\
   --weights results/.../best_model.pth \\
   --dataset loreal --sequences txts/my_seqs.txt \\
   --data-scale 255 --n-b 8


 # FMDD: validate PURE against true MSE
 python compute_pure.py --model drunet \\
   --weights results/.../best_model.pth \\
   --dataset fmdd --fmdd-mode clean \\
   --sequences txts/my_seqs.txt --n-b 8


 # Generic TIF sequences (blind)
 python compute_pure.py --model drunet \\
   --weights results/.../best_model.pth \\
   --dataset tif-seq --sequences txts/my_seqs.txt \\
   --data-scale 65535 --n-b 8
"""


from __future__ import annotations


import argparse
import csv
from pathlib import Path


import deepinv as dinv
import numpy as np
import torch
from tqdm import tqdm


from dataset import LorealSequenceDataset
from inference import (
   apply_params_from_checkpoint_dir,
   build_model,
   compute_pure_mc,
   crop_to_model,
   device,
   load_weights,
   read_png_tensor,
   read_sequence_lines,
   read_tif_tensor,
   resolve_fmdd_sequence,
   resolve_loreal_sequence,
   resolve_tif_sequence,
)

# ---------------------------------------------------------------------------
# Per-sequence PURE functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def pure_loreal_sequence(
   model: torch.nn.Module,
   model_type: str,
   physics,
   seq_path: str,
   a: float,
   b_param: float,
   data_scale: float,
   noise_level: float,
   max_frames: int | None,
   frame_stride: int,
   n_b: int,
) -> dict:
   seq_name = Path(seq_path).name
   frame_pures: list[float] = []


   if model_type == "drunet":
       tif_files = sorted(Path(seq_path).glob("*.tif"))
       indices = list(range(0, len(tif_files), frame_stride))
       if max_frames is not None:
           indices = indices[:max_frames]
       for i in tqdm(indices, desc=seq_name, leave=False):
           y = read_tif_tensor(tif_files[i], data_scale).unsqueeze(0).to(device)
           y = torch.clamp(y, min=0.0)
           y = crop_to_model(y, model_type)
           samples = [compute_pure_mc(model, y, physics, noise_level) for _ in range(n_b)]
           frame_pures.append(float(np.mean(samples)))


   else:  # fastdvdnet (for now)
       seq_ds = LorealSequenceDataset(
           sequence_info=[(seq_path, a, b_param)],
           num_frames=5,
           data_scale=data_scale,
       )
       if len(seq_ds) == 0:
           print(f"  SKIP {seq_name}: no valid 5-frame windows")
           return {}
       indices = list(range(0, len(seq_ds), frame_stride))
       if max_frames is not None:
           indices = indices[:max_frames]
       for i in tqdm(indices, desc=seq_name, leave=False):
           y_stack, y_central = seq_ds[i]
           y_stack = crop_to_model(y_stack.unsqueeze(0).to(device), model_type)
           y_central = y_stack[:, y_stack.shape[1] // 2 : y_stack.shape[1] // 2 + 1]
           samples = [
               compute_pure_mc(model, y_central, physics, noise_level, y_stack)
               for _ in range(n_b)
           ]
           frame_pures.append(float(np.mean(samples)))


   if not frame_pures:
       return {}
   return {
       "sequence": seq_name,
       "n_frames": len(frame_pures),
       "pure_mean": float(np.mean(frame_pures)),
       "pure_std": float(np.std(frame_pures)),
   }




@torch.no_grad()
def pure_fmdd_clean(
   model: torch.nn.Module,
   model_type: str,
   physics,
   seq: dict,
   noise_level: float,
   n_b: int,
) -> dict | None:
   """FMDD with synthetic Poisson noise. Returns PURE + true MSE for validation."""
   gt_path = seq.get("gt")
   if not gt_path or not Path(gt_path).exists():
       print(f"  SKIP {seq['modality']}/{seq['seq_id']}: no GT (avg50.png)")
       return None


   x_clean = read_png_tensor(Path(gt_path)).unsqueeze(0).to(device)
   x_clean = crop_to_model(x_clean, model_type)


   if model_type == "drunet":
       y = physics(x_clean)
       x_est = model(y, physics)
       samples = [compute_pure_mc(model, y, physics, noise_level) for _ in range(n_b)]
   else:
       noisy_frames = [physics(x_clean) for _ in range(5)]
       y_stack = crop_to_model(torch.cat(noisy_frames, dim=1), model_type)
       y_central = y_stack[:, 2:3]
       model.model.set_context(y_stack)
       x_est = model(y_central, physics)
       samples = [
           compute_pure_mc(model, y_central, physics, noise_level, y_stack)
           for _ in range(n_b)
       ]


   true_mse = ((x_est - x_clean) ** 2).mean().item()
   return {
       "sequence": f"{seq['modality']}/{seq['seq_id']}",
       "n_frames": 1,
       "pure_mean": float(np.mean(samples)),
       "pure_std": float(np.std(samples)),
       "true_mse": true_mse,
   }




@torch.no_grad()
def pure_fmdd_raw(
   model: torch.nn.Module,
   model_type: str,
   physics,
   seq: dict,
   noise_level: float,
   max_frames: int | None,
   frame_stride: int,
   n_b: int,
) -> dict:
   """FMDD real noisy PNGs. Blind PURE (no GT needed)."""
   frames = seq["frames"]
   tag = f"{seq['modality']}/{seq['seq_id']}"
   frame_pures: list[float] = []


   if model_type == "drunet":
       indices = list(range(0, len(frames), frame_stride))
       if max_frames is not None:
           indices = indices[:max_frames]
       for i in tqdm(indices, desc=tag, leave=False):
           y = read_png_tensor(Path(frames[i])).unsqueeze(0).to(device)
           y = crop_to_model(y, model_type)
           samples = [compute_pure_mc(model, y, physics, noise_level) for _ in range(n_b)]
           frame_pures.append(float(np.mean(samples)))
   else:
       mid = 2
       if len(frames) < 5:
           print(f"  SKIP {tag}: fewer than 5 frames for FastDVDNet")
           return {}
       center_indices = list(range(mid, len(frames) - mid, frame_stride))
       if max_frames is not None:
           center_indices = center_indices[:max_frames]
       for i in tqdm(center_indices, desc=tag, leave=False):
           window = [
               read_png_tensor(Path(frames[j])).unsqueeze(0).to(device)
               for j in range(i - mid, i + mid + 1)
           ]
           y_stack = crop_to_model(torch.cat(window, dim=1), model_type)
           y_central = y_stack[:, 2:3]
           samples = [
               compute_pure_mc(model, y_central, physics, noise_level, y_stack)
               for _ in range(n_b)
           ]
           frame_pures.append(float(np.mean(samples)))


   if not frame_pures:
       return {}
   return {
       "sequence": tag,
       "n_frames": len(frame_pures),
       "pure_mean": float(np.mean(frame_pures)),
       "pure_std": float(np.std(frame_pures)),
   }




@torch.no_grad()
def pure_tif_sequence(
   model: torch.nn.Module,
   model_type: str,
   physics,
   tif_files: list[Path],
   data_scale: float,
   noise_level: float,
   max_frames: int | None,
   frame_stride: int,
   n_b: int,
   gt_path: Path | None = None,
) -> dict:
   """Generic TIF sequence. Pass gt_path to also compute true MSE alongside PURE."""
   seq_name = tif_files[0].parent.name
   frame_pures: list[float] = []
   frame_mses: list[float] = []


   # Load GT once — it's the same clean image for every noisy frame in the sequence
   x_clean = None
   if gt_path is not None:
       if not gt_path.exists():
           print(f"  WARNING: GT not found at {gt_path}, skipping true MSE")
       else:
        #    x_clean = read_tif_tensor(gt_path, data_scale).unsqueeze(0).to(device)
        #    x_clean = torch.clamp(x_clean, min=0.0)
        #    x_clean = crop_to_model(x_clean, model_type)
        x_clean = read_png_tensor(gt_path).unsqueeze(0).to(device)
        x_clean = torch.clamp(x_clean, min=0.0)
        x_clean = crop_to_model(x_clean, model_type)


   if model_type == "drunet":
       indices = list(range(0, len(tif_files), frame_stride))
       if max_frames is not None:
           indices = indices[:max_frames]
       for i in tqdm(indices, desc=seq_name, leave=False):
           y = read_tif_tensor(tif_files[i], data_scale).unsqueeze(0).to(device)
           y = torch.clamp(y, min=0.0)
           y = crop_to_model(y, model_type)
           samples = [compute_pure_mc(model, y, physics, noise_level) for _ in range(n_b)]
           frame_pures.append(float(np.mean(samples)))
           if x_clean is not None:
               x_est = model(y, physics)
               frame_mses.append(((x_est - x_clean) ** 2).mean().item())
   else:
       mid = 2
       if len(tif_files) < 5:
           print(f"  SKIP {seq_name}: fewer than 5 TIF files for FastDVDNet")
           return {}
       center_indices = list(range(mid, len(tif_files) - mid, frame_stride))
       if max_frames is not None:
           center_indices = center_indices[:max_frames]
       for i in tqdm(center_indices, desc=seq_name, leave=False):
           window = [
               read_tif_tensor(tif_files[j], data_scale).unsqueeze(0).to(device)
               for j in range(i - mid, i + mid + 1)
           ]
           y_stack = torch.clamp(torch.cat(window, dim=1), min=0.0)
           y_stack = crop_to_model(y_stack, model_type)
           y_central = y_stack[:, 2:3]
           samples = [
               compute_pure_mc(model, y_central, physics, noise_level, y_stack)
               for _ in range(n_b)
           ]
           frame_pures.append(float(np.mean(samples)))
           if x_clean is not None:
               model.model.set_context(y_stack)
               x_est = model(y_central, physics)
               frame_mses.append(((x_est - x_clean) ** 2).mean().item())


   if not frame_pures:
       return {}
   result = {
       "sequence": seq_name,
       "n_frames": len(frame_pures),
       "pure_mean": float(np.mean(frame_pures)),
       "pure_std": float(np.std(frame_pures)),
       "frame_pures": frame_pures,
   }
   if frame_mses:
       result["true_mse_mean"] = float(np.mean(frame_mses))
       result["true_mse_std"] = float(np.std(frame_mses))
       result["frame_mses"] = frame_mses
   return result

def write_per_frame_csv(row: dict, output_dir: Path, run_name: str) -> None:
    seq_name = row["sequence"].replace("/", "_")
    out_path = output_dir / f"{seq_name}_{run_name}_frames.csv"
    frame_pures = row.get("frame_pures", [])
    frame_mses  = row.get("frame_mses", [])
    has_mse = bool(frame_mses)
    fieldnames = ["frame", "pure"] + (["true_mse"] if has_mse else [])
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fi, pure_val in enumerate(frame_pures):
            entry = {"frame": fi+2, "pure": pure_val}
            if has_mse:
                entry["true_mse"] = frame_mses[fi] if fi < len(frame_mses) else ""
            writer.writerow(entry)
    print(f"    Per-frame CSV: {out_path}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------




def main() -> None:
   parser = argparse.ArgumentParser(
       description="Compute PURE (Poisson Unbiased Risk Estimator) — blind MSE estimate."
   )
   parser.add_argument("--model", choices=["drunet", "fastdvdnet"], required=True)
   parser.add_argument("--weights", type=Path, required=True, help="Path to .pth checkpoint")
   parser.add_argument("--dataset", choices=["fmdd", "loreal", "tif-seq"], required=True)
   parser.add_argument("--sequences", type=Path, required=True, help="Text file with one sequence per line")
   parser.add_argument("--output", type=Path, default=None, help="Output CSV path (default: pure_<dataset>_<model>.csv)")
   parser.add_argument("--fmdd-root", type=Path, default=Path("../data/FMDD"))
   parser.add_argument(
       "--loreal-data-dir",
       type=Path,
       default=Path("/home/diegosilvera/Escritorio/2026/sequences_almost_Poisson"),
   )
   parser.add_argument(
       "--fmdd-mode",
       choices=["clean", "raw"],
       default="raw",
       help="'clean' adds true MSE column for validation; 'raw' is blind",
   )
   parser.add_argument(
       "--noise",
       type=float,
       default=None,
       help="Poisson noise level = 1/data_scale for γ=1 data (default: derived from --data-scale)",
   )
   parser.add_argument("--alpha", type=float, default=0.15, help="R2R alpha used at training time")
   parser.add_argument("--data-scale", type=float, default=None, help="Pixel scale factor (required for tif-seq)")
   parser.add_argument(
       "--n-b",
       type=int,
       default=8,
       help="Rademacher samples per frame (more → lower variance, costs 2 forward passes each)",
   )
   parser.add_argument(
       "--gt-dir",
       type=Path,
       default=None,
       help="Directory with GT TIF files named <sequence_name>.png (one per sequence). "
            "Only used with --dataset tif-seq. Adds true_mse_mean/std columns to the CSV.",
   )
   parser.add_argument("--max-frames", type=int, default=None)
   parser.add_argument("--frame-stride", type=int, default=1)
   parser.add_argument(
       "--use-params-from-checkpoint-dir",
       action="store_true",
       help="Load noise/alpha from parameters.txt next to the weights file",
   )
   args = parser.parse_args()


   if not args.weights.exists():
       raise FileNotFoundError(f"Weights not found: {args.weights}")
   if args.use_params_from_checkpoint_dir:
       apply_params_from_checkpoint_dir(args)


   # Resolve data_scale and noise_level
   if args.dataset == "fmdd":
       data_scale = 1.0  # read_png_tensor normalizes to [0,1] internally
       noise = args.noise if args.noise is not None else 1 / 255.0
   elif args.dataset == "loreal":
       data_scale = args.data_scale if args.data_scale is not None else 255.0
       noise = args.noise if args.noise is not None else 1.0 / data_scale
   else:  # tif-seq
       if args.data_scale is None:
           raise ValueError("--data-scale required for --dataset tif-seq (e.g. 255 or 65535)")
       data_scale = args.data_scale
       noise = args.noise if args.noise is not None else 1.0 / data_scale


   noise_level = noise  # for γ=1 pure Poisson: noise_level = 1/data_scale


   output_path = args.output or Path(f"pure_{args.dataset}_{args.model}.csv")


   lines = read_sequence_lines(args.sequences)
   print(
       f"{len(lines)} sequence(s) | model={args.model} | device={device} | "
       f"noise_level={noise_level:.2e} | n_b={args.n_b}"
   )


   model = build_model(args.model, noise, args.alpha)
   load_weights(model, args.weights)
   model.eval()


   noise_model_obj = dinv.physics.PoissonNoise(noise)
   noise_model_obj.sigma = noise
   physics = dinv.physics.Denoising(noise_model=noise_model_obj)


   rows: list[dict] = []


   if args.dataset == "loreal":
       seqs = [resolve_loreal_sequence(line, args.loreal_data_dir) for line in lines]
       for seq_path, a, b_param in seqs:
           print(f"  {Path(seq_path).name}...")
           row = pure_loreal_sequence(
               model, args.model, physics, seq_path, a, b_param,
               data_scale, noise_level, args.max_frames, args.frame_stride, args.n_b,
           )
           if row:
               rows.append(row)
               print(f"    DEBUG line 470 row keys: {list(row.keys())}")
               write_per_frame_csv(row, output_path.parent, output_path.stem)
               print(f"    PURE = {row['pure_mean']:.4e} ± {row['pure_std']:.4e}  ({row['n_frames']} frames)")


   elif args.dataset == "fmdd":
       fmdd_seqs = [resolve_fmdd_sequence(line, args.fmdd_root) for line in lines]
       for seq in fmdd_seqs:
           label = f"{seq['modality']}/{seq['seq_id']}"
           print(f"  {label} ({args.fmdd_mode})...")
           if args.fmdd_mode == "clean":
               row = pure_fmdd_clean(model, args.model, physics, seq, noise_level, args.n_b)
           else:
               row = pure_fmdd_raw(
                   model, args.model, physics, seq,
                   noise_level, args.max_frames, args.frame_stride, args.n_b,
               )
           if row:
               rows.append(row)
               print(f"    DEBUG line 488 row keys: {list(row.keys())}")
               write_per_frame_csv(row, output_path.parent, output_path.stem)
               msg = f"    PURE = {row['pure_mean']:.4e} ± {row['pure_std']:.4e}"
               if "true_mse" in row:
                   msg += f"  |  true MSE = {row['true_mse']:.4e}"
               print(msg)


   else:  # tif-seq
       tif_seqs = [resolve_tif_sequence(line) for line in lines]
       for tif_files in tif_seqs:
           seq_name = tif_files[0].parent.name
           gt_path = (args.gt_dir / f"{seq_name}.png") if args.gt_dir is not None else None
           print(f"  {seq_name} ({len(tif_files)} frames)...")
           row = pure_tif_sequence(
               model, args.model, physics, tif_files,
               data_scale, noise_level, args.max_frames, args.frame_stride, args.n_b,
               gt_path=gt_path,
           )
           if row:
               rows.append(row)
               print(f"    DEBUG line 510 row keys: {list(row.keys())}")
               print(f"    DEBUG frame_pures: {len(row.get('frame_pures', []))} frames, frame_mses: {len(row.get('frame_mses', []))} frames")
               write_per_frame_csv(row, output_path.parent, output_path.stem)
               msg = f"    PURE = {row['pure_mean']:.4e} ± {row['pure_std']:.4e}"
               if "true_mse_mean" in row:
                   msg += f"  |  true MSE = {row['true_mse_mean']:.4e} ± {row['true_mse_std']:.4e}"
               print(msg)


   if not rows:
       print("No results computed.")
       return


   # Collect union of all keys (rows may differ if some sequences were missing GT)
   fieldnames: list[str] = []
   for r in rows:
       for k in r:
           if k not in fieldnames:
               fieldnames.append(k)
   with open(output_path, "w", newline="") as f:
       writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
       writer.writeheader()
       writer.writerows(rows)


   mean_pure = float(np.mean([r["pure_mean"] for r in rows]))
   print(f"\nMean PURE: {mean_pure:.4e}  |  Results saved to {output_path}")




if __name__ == "__main__":
   main()




