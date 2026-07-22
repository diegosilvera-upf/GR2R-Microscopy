"""Plotting helpers for GR2R training runs."""

from pathlib import Path

import matplotlib.pyplot as plt


def save_loss_plot(train_losses, val_losses, val_psnrs, cfg, output_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, label="Train Loss")
    ax1.plot(epochs_range, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    if val_psnrs:
        ax2 = ax1.twinx()
        ax2.plot(epochs_range, val_psnrs, "g--", label="Val PSNR (dB)")
        ax2.set_ylabel("Val PSNR (dB)")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
    else:
        ax1.legend()

    title = f"{cfg.project_name} — {cfg.model} / {cfg.dataset}"
    ax1.set_title(title)
    path = output_dir / "loss_plot.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Loss plot saved to {path}")
