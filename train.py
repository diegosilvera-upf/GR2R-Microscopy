r"""
Unified GR2R training script. All hyperparameters are defined in a YAML config file.

Usage:
  python train.py --config configs/fmdd_drunet.yaml
  python train.py --config configs/loreal_fastdvdnet.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import deepinv as dinv
import torch
import torch.optim as optim
from tqdm import tqdm

from config import load_config
from export_utils import export_fmdd_samples, export_loreal_sequences
from training_utils import (
    build_eval_cache,
    build_fmdd_datasets,
    build_loreal_datasets,
    build_train_model,
    device,
    evaluate_fmdd,
    evaluate_loreal,
    load_pretrained,
    save_config,
)
from viz_utils import save_loss_plot

BASE_DIR = Path(".")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GR2R unified training.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.model in ("drunet", "fastdvdnet"), f"Unknown model: {cfg.model}"
    assert cfg.dataset in ("fmdd", "loreal"), f"Unknown dataset: {cfg.dataset}"
    assert cfg.loss in ("r2r", "l2r"), f"Unknown loss: {cfg.loss}"

    # Output directory
    if cfg.inference_dir is not None:
        output_dir = Path(cfg.inference_dir)
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        results_dir = BASE_DIR / "results" / cfg.project_name
        results_dir.mkdir(parents=True, exist_ok=True)
        output_dir = results_dir / f"tif_output_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir, args.config)

    print(f"Device: {device} | model: {cfg.model} | dataset: {cfg.dataset}")
    print(f"Output: {output_dir}")

    # Physics
    noise_model = dinv.physics.PoissonNoise(cfg.gamma)
    noise_model.sigma = cfg.gamma
    physics = dinv.physics.Denoising(noise_model=noise_model)

    # Model
    model, criterion = build_train_model(cfg, noise_model)
    load_pretrained(model, cfg)
    if cfg.loss == "l2r":
        optimizer = optim.Adam(model.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.epochs * 0.8) + 1)

    # If inference_dir is set: skip training, just export
    if cfg.inference_dir is not None:
        print("inference_dir set — skipping training, running export only.")
        if cfg.dataset == "fmdd":
            _, _, test_dataset, test_seq, visualize_indices = build_fmdd_datasets(cfg)
            export_fmdd_samples(model, cfg, physics, test_dataset, visualize_indices, output_dir)
        else:
            _, _, _, val_seq, test_seq, visualize_names = build_loreal_datasets(cfg)
            explicit_viz = [s for s in val_seq + test_seq if Path(s[0]).name in set(visualize_names)]
            seqs = explicit_viz or val_seq
            export_loreal_sequences(model, cfg, physics, seqs, output_dir)
        print(f"Done. Outputs in {output_dir}")
        return

    # Datasets
    num_frames = 5 if cfg.model == "fastdvdnet" else 1

    if cfg.dataset == "fmdd":
        train_loader, train_dataset, test_dataset, test_seq, visualize_indices = build_fmdd_datasets(cfg)
        n_eval = len(test_seq)
        if cfg.n_eval_sequences is not None:
            n_eval = min(n_eval, cfg.n_eval_sequences)
        eval_cache = build_eval_cache(test_dataset, physics, n_eval, device, cfg.eval_seed, num_frames=num_frames)
        val_loader = None
        val_seq = None
        visualize_names = None
    else:
        train_loader, val_loader, train_seq, val_seq, test_seq, visualize_names = build_loreal_datasets(cfg)
        eval_cache = None
        test_dataset = None
        visualize_indices = None

    # Checkpoint tracking — FMDD keeps best-by-loss AND best-by-PSNR; Loreal only best-by-loss
    best_val_loss = float("inf")
    best_loss_epoch = -1
    best_ckpt_loss = output_dir / "best_model_loss.pth"

    best_psnr = 0.0
    best_psnr_epoch = -1
    best_ckpt_psnr = output_dir / "best_model_psnr.pth"  # only used for FMDD

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_psnrs: list[float] = []  # empty for Loreal

    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        running_train = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            # Prepare inputs
            if cfg.dataset == "fmdd":
                x_clean_stack, _ = batch
                x_clean_stack = x_clean_stack.to(device)
                if cfg.model == "drunet":
                    y = physics(x_clean_stack[:, 0:1])
                else:
                    noisy_frames = [physics(x_clean_stack[:, j:j+1]) for j in range(5)]
                    y = torch.cat(noisy_frames, dim=1)
            else:  # loreal — data already noisy
                y_stack, _ = batch
                y = y_stack.to(device)

            # Forward pass
            if cfg.model == "drunet":
                x_est = model(y, physics, update_parameters=True)
                loss = criterion(x_est, y, physics, model)
            else:
                y_central = y[:, 2:3]
                model.model.set_context(y)
                x_est = model(y_central, physics, update_parameters=True)
                loss = criterion(x_est, y_central, physics, model)

            loss.backward()
            optimizer.step()
            running_train += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        train_loss_epoch = running_train / max(len(train_loader), 1)

        # Evaluation
        if cfg.dataset == "fmdd":
            current_psnr, current_val_loss = evaluate_fmdd(model, cfg, physics, criterion, eval_cache)
            val_psnrs.append(current_psnr)
            print(
                f"Epoch {epoch+1}: train_loss={train_loss_epoch:.6f} | "
                f"val_loss={current_val_loss:.6f} | val_psnr={current_psnr:.2f} dB"
            )
        else:
            current_val_loss = evaluate_loreal(model, cfg, val_loader, physics, criterion)
            current_psnr = None
            print(
                f"Epoch {epoch+1}: train_loss={train_loss_epoch:.6f} | "
                f"val_loss={current_val_loss:.6f}"
            )

        train_losses.append(train_loss_epoch)
        val_losses.append(current_val_loss)

        # Checkpoint by val_loss
        is_best = (
            abs(current_val_loss) < abs(best_val_loss)
            if cfg.loss == "l2r"
            else current_val_loss < best_val_loss
        )
        if is_best:
            best_val_loss = current_val_loss
            best_loss_epoch = epoch + 1
            torch.save(model.state_dict(), best_ckpt_loss)
            print(f"  New best val_loss model saved (epoch={best_loss_epoch})")

        # Checkpoint by PSNR (FMDD only)
        if current_psnr is not None and current_psnr > best_psnr:
            best_psnr = current_psnr
            best_psnr_epoch = epoch + 1
            torch.save(model.state_dict(), best_ckpt_psnr)
            print(f"  New best PSNR model saved (epoch={best_psnr_epoch}, {best_psnr:.2f} dB)")

        scheduler.step()

        # Periodic checkpoint
        if (epoch + 1) % cfg.checkpoint_every == 0:
            periodic_path = output_dir / f"model_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), periodic_path)
            print(f"  Periodic checkpoint: {periodic_path}")

    # Save training summary
    with open(output_dir / "best_checkpoint.txt", "w") as f:
        f.write(f"best_loss_epoch={best_loss_epoch}\n")
        f.write(f"best_val_loss={best_val_loss:.6f}\n")
        f.write(f"weights_loss_path={best_ckpt_loss}\n")
        if cfg.dataset == "fmdd":
            f.write(f"best_psnr_epoch={best_psnr_epoch}\n")
            f.write(f"best_val_psnr={best_psnr:.4f}\n")
            f.write(f"weights_psnr_path={best_ckpt_psnr}\n")

    save_loss_plot(train_losses, val_losses, val_psnrs if val_psnrs else None, cfg, output_dir)

    # Export results with best weights
    if cfg.dataset == "fmdd":
        for ckpt_path, epoch_n, tag in [
            (best_ckpt_psnr, best_psnr_epoch, "psnr"),
            (best_ckpt_loss, best_loss_epoch, "loss"),
        ]:
            if ckpt_path.exists():
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"Exporting with best_{tag} weights (epoch {epoch_n})...")
                export_fmdd_samples(model, cfg, physics, test_dataset, visualize_indices, output_dir, tag=tag)
    else:
        if best_ckpt_loss.exists():
            model.load_state_dict(torch.load(best_ckpt_loss, map_location=device))
        explicit_viz = [s for s in val_seq + test_seq if Path(s[0]).name in set(visualize_names)]
        if explicit_viz:
            export_loreal_sequences(model, cfg, physics, explicit_viz, output_dir, tag="viz")
        else:
            export_loreal_sequences(model, cfg, physics, val_seq, output_dir, tag="val")
            export_loreal_sequences(model, cfg, physics, test_seq, output_dir, tag="test")

    print(f"Finished. Check results in {output_dir}")


if __name__ == "__main__":
    main()
