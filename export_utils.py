"""Export helpers for GR2R training runs — dump denoised samples/sequences to disk."""

from pathlib import Path

import numpy as np
import tifffile
import torch
from tqdm import tqdm

from dataset import LorealSequenceDataset
from training_utils import device


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

                denoised_frames.append((x_est * cfg.data_scale).squeeze().cpu().numpy().astype(np.float32))
                noisy_frames.append((y_target * cfg.data_scale).squeeze().cpu().numpy().astype(np.float32))

            safe_name = seq_name.replace("/", "_")
            tifffile.imwrite(str(output_dir / f"{suffix}{safe_name}_denoised.tif"), np.stack(denoised_frames, axis=0))
            tifffile.imwrite(str(output_dir / f"{suffix}{safe_name}_noisy.tif"), np.stack(noisy_frames, axis=0))
            print(f"  Saved {safe_name} ({len(denoised_frames)} frames).")
