"""YAML config loading for GR2R training runs."""

from types import SimpleNamespace

import yaml


def load_config(yaml_path: str) -> SimpleNamespace:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    defaults = {
        "pretrained_ckpt": None,
        "inference_dir": None,
        "n_eval_sequences": None,
        "checkpoint_every": 10,
        "max_export_sequences": 3,
        "num_workers": 4,
        "repeats_per_sequence": 55,
        "repeats_per_frame": 10,
        "val_batch_size": None,
        "val_prefixes": [],
        "test_prefixes": [],
        "fmdd_data_dir": "../data/FMDD",
        "fmdd_split_file": "txts/fmdd_split.txt",
        "fmdd_modalities": None,
        "loreal_data_dir": None,
        "loreal_split_file": "txts/loreal_split.txt",
        "data_scale": None,
        "eval_seed": 42,
        "loss": "r2r",
        "l2r_eval_n_samples": 5,
        "l2r_recorruptor_ckpt": None,
        "l2r_recorruptor_lr": 1e-4,
    }
    for k, v in defaults.items():
        raw.setdefault(k, v)

    if raw["data_scale"] is None:
        raw["data_scale"] = 1.0 if raw["dataset"] == "fmdd" else 255.0
    if raw["val_batch_size"] is None:
        raw["val_batch_size"] = raw["batch_size"]

    return SimpleNamespace(**raw)
