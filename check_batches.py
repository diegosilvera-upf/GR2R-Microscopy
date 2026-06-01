
import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np

# Setup paths
BASE_DIR = Path("/home/diegosilvera/Escritorio/GeneralizedR2R")
DEEPINV_PATH = BASE_DIR / "GeneralizedR2R" / "deepinv"
sys.path.append(str(DEEPINV_PATH))

import deepinv as dinv
from loreal_dataset_fixed import FMDDDataset, get_fmdd_sequences

def check():
    noise_level = 1/255.0
    ORIGINAL_DATA_DIR = Path("/home/diegosilvera/Escritorio/data/FMDD")
    img_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B', 'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R', 'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH', 'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
    
    sequences = get_fmdd_sequences(ORIGINAL_DATA_DIR, modalities=img_types)
    n_total = len(sequences)
    n_train = int(11 * n_total / 12)
    train_seq = sequences[:n_train]
    
    train_dataset = FMDDDataset(sequence_info=train_seq, patch_size=(128, 128), 
                                mode='synthetic', gamma=1.0/noise_level, data_scale=1.0, 
                                num_frames=1)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    print(f"Total sequences: {len(sequences)}")
    print(f"Train sequences: {len(train_seq)}")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total training batches: {len(train_dataloader)}")

if __name__ == "__main__":
    check()
