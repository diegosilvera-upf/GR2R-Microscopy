import torch
import numpy as np
import tifffile
from pathlib import Path
from torch.utils.data import Dataset
import imageio.v3 as iio

# ----------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------
def linear_transform(x, a, b, u=1):
    """Linear transform from pre-processing.txt"""
    return a * x + u * b

# ----------------------------------------------------------------------------------
# Loreal Dataset Discovery and Classes
# ----------------------------------------------------------------------------------

def get_valid_sequences(sequence_paths, out_file="sequences_left_out_local.txt"):
    """
    Filters sequences based on pre-processing.txt and minimum frame count.
    """
    valid_sequences = []
    with open(out_file, "w") as f_out:
        for seq in sequence_paths:
            seq = Path(seq)
            if not seq.is_dir():
                continue
            
            preproc_file = seq / "pre-processing.txt"
            if not preproc_file.exists():
                f_out.write(f"{seq.name}, no pre-processing.txt file\n")
                continue
            
            try:
                # Some files have a and b separated by space or newline
                params = np.loadtxt(preproc_file)
                if params.ndim == 1:
                    a, b = params[0], params[1]
                else:
                    a, b = params.flatten()[0], params.flatten()[1]
            except Exception as e:
                f_out.write(f"{seq.name}, error reading pre-processing.txt: {e}\n")
                continue

            if np.abs(a-1) > 0.2:
                f_out.write(f"{seq.name}, a={a}\n")
                continue
            
            tif_files = sorted(seq.glob("*.tif"))
            if not tif_files:
                continue
            
            # Check for enough frames in at least one channel
            names = [f.name for f in tif_files]
            channels = ["_c0_", "_c1_"] if any("_c0_" in n or "_c1_" in n for n in names) else [""]
            
            has_enough_frames = False
            for ch in channels:
                frames = [f for f in tif_files if ch in f.name] if ch else tif_files
                if len(frames) >= 5:
                    has_enough_frames = True
                    break
            
            if has_enough_frames:
                valid_sequences.append((str(seq), float(a), float(b)))
            else:
                f_out.write(f"{seq.name}, not enough frames (min 5)\n")
                
    return sorted(valid_sequences)

class LorealSequenceDataset(Dataset):
    """
    Localized version of Loreal dataset that supports both single-frame (DRUNet)
    and multi-frame (FastDVDNet) loading from the same sequence structure.
    """
    def __init__(self, sequence_info, patch_size=None, transform=None, data_scale=255.0, num_frames=5):
        self.patch_size = patch_size
        self.transform = transform
        self.data_scale = data_scale
        self.num_frames = num_frames
        self.stacks = []

        for seq_path, a, b in sequence_info:
            seq = Path(seq_path)
            tif_files = sorted(seq.glob("*.tif"))
            names = [f.name for f in tif_files]
            channels = ["_c0_", "_c1_"] if any("_c0_" in n or "_c1_" in n for n in names) else [""]
            
            for ch in channels:
                frames = sorted(f for f in tif_files if ch in f.name) if ch else sorted(tif_files)
                if len(frames) < self.num_frames:
                    continue
                
                mid = self.num_frames // 2
                for i in range(mid, len(frames) - mid):
                    stack_paths = [str(f) for f in frames[i-mid : i-mid+self.num_frames]]
                    self.stacks.append((stack_paths, float(a), float(b)))

    def __len__(self):
        return len(self.stacks)

    def _read_tif(self, path):
        img = iio.imread(str(path)).astype(np.float32)
        img = torch.from_numpy(img)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3:
            img = img.permute(2, 0, 1).mean(dim=0, keepdim=True)
        return img

    def make_divisible_by_4(self, img):
        H, W = img.shape[-2:]
        H4, W4 = (H // 4) * 4, (W // 4) * 4
        return img[..., :H4, :W4]

    def __getitem__(self, idx):
        stack_paths, a, b = self.stacks[idx]
        frames = [self._read_tif(p) for p in stack_paths]
        stack = torch.cat(frames, dim=0)
        
        stack = self.make_divisible_by_4(stack)
        stack = linear_transform(stack, a, b, u=1) / self.data_scale
        stack = torch.clamp(stack, min=0.0)

        if self.patch_size is not None:
            H, W = stack.shape[-2:]
            ph, pw = self.patch_size
            top = torch.randint(0, H - ph + 1, (1,)).item()
            left = torch.randint(0, W - pw + 1, (1,)).item()
            stack = stack[:, top:top+ph, left:left+pw]

        if self.transform:
            stack = self.transform(stack)
            
        target = stack[self.num_frames // 2 : self.num_frames // 2 + 1, :, :].clone()
        return stack, target

# ----------------------------------------------------------------------------------
# FMDD Dataset Discovery and Classes
# ----------------------------------------------------------------------------------

def get_fmdd_sequences(root_dir, modalities=None):
    root = Path(root_dir)
    sequences = []
    if modalities is None:
        modalities = [d.name for d in root.iterdir() if d.is_dir()]
    
    for mod in modalities:
        mod_raw = root / mod / "raw"
        if not mod_raw.exists(): continue
        for seq_dir in mod_raw.iterdir():
            if not seq_dir.is_dir(): continue
            png_files = sorted(seq_dir.glob("*.png"))
            if len(png_files) >= 5:
                gt_path = root / mod / "gt" / seq_dir.name / "avg50.png"
                sequences.append({
                    'modality': mod,
                    'seq_id': seq_dir.name,
                    'frames': [str(p) for p in png_files],
                    'gt': str(gt_path) if gt_path.exists() else None
                })
    return sequences

class FMDDDataset(Dataset):
    def __init__(self, sequence_info, patch_size=None, transform=None, data_scale=255.0, mode='raw', gamma=None, num_frames=5):
        self.patch_size = patch_size
        self.transform = transform
        self.data_scale = data_scale
        self.mode = mode
        self.gamma = gamma
        self.num_frames = num_frames
        self.stacks = []

        for seq in sequence_info:
            frames, gt = seq['frames'], seq['gt']
            if self.mode == 'raw':
                if len(frames) >= self.num_frames:
                    mid = self.num_frames // 2
                    for i in range(mid, len(frames) - mid):
                        stack_paths = frames[i-mid : i-mid+self.num_frames]
                        self.stacks.append((stack_paths, gt))
            elif self.mode == 'synthetic':
                if gt:
                    for _ in range(10): # 10 stacks per sequences to increase dataset size
                        self.stacks.append((None, gt))

    def __len__(self):
        return len(self.stacks)

    def _read_png(self, path):
        img = iio.imread(str(path)).astype(np.float32)
        img = torch.from_numpy(img)
        if img.ndim == 2: img = img.unsqueeze(0)
        elif img.ndim == 3: img = img.permute(2, 0, 1).mean(dim=0, keepdim=True)
        return img

    def _add_poisson_noise(self, img):
        if self.gamma is None: return img
        img = torch.clamp(img, min=0.0)
        return torch.poisson(img * self.gamma) / self.gamma

    def make_divisible_by_4(self, img):
        H, W = img.shape[-2:]
        H4, W4 = (H // 4) * 4, (W // 4) * 4
        return img[..., :H4, :W4]

    def __getitem__(self, idx):
        stack_paths, gt_path = self.stacks[idx]
        
        if self.mode == 'raw':
            frames = [self._read_png(p) for p in stack_paths]
            stack = torch.cat(frames, dim=0)
            if gt_path:
                target_full = self._read_png(gt_path)
            else:
                target_full = stack[self.num_frames // 2 : self.num_frames // 2 + 1, :, :].clone()
            
            if self.patch_size is not None:
                H, W = stack.shape[-2:]
                ph, pw = self.patch_size
                top = torch.randint(0, H - ph + 1, (1,)).item()
                left = torch.randint(0, W - pw + 1, (1,)).item()
                stack = stack[:, top:top+ph, left:left+pw]
                target = target_full[:, top:top+ph, left:left+pw]
            else:
                target = target_full
        else: # synthetic
            gt_full = self._read_png(gt_path)
            if self.patch_size is not None:
                H, W = gt_full.shape[-2:]
                ph, pw = self.patch_size
                top = torch.randint(0, H - ph + 1, (1,)).item()
                left = torch.randint(0, W - pw + 1, (1,)).item()
                gt_patch = gt_full[:, top:top+ph, left:left+pw]
            else:
                gt_patch = gt_full
            
            frames = [self._add_poisson_noise(gt_patch) for _ in range(self.num_frames)]
            stack = torch.cat(frames, dim=0)
            target = gt_patch

        stack = self.make_divisible_by_4(stack) / self.data_scale
        target = self.make_divisible_by_4(target) / self.data_scale
        
        if self.transform:
            combined = torch.cat([stack, target], dim=0)
            combined = self.transform(combined)
            stack = combined[:self.num_frames, :, :]
            target = combined[self.num_frames:, :, :]
            
        return stack, target
