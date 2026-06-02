from torch.utils.data import Dataset
import tifffile
import torch
import numpy as np

from pathlib import Path
from torchvision.io import read_image
from loreal_utils import *
import imageio.v3 as iio

class LorealDataset(Dataset):
    def __init__(self, image_paths, transform=None, patch_size=None):
        self.image_paths = image_paths
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = tifffile.imread(self.image_paths[idx]).astype(np.float32)

        img = torch.from_numpy(img)

        if self.patch_size:
            img = self.random_crop(img)

        if self.transform:
            img = self.transform(img)

        return img

    def random_crop(self, img):
        H, W = img.shape
        ph, pw = self.patch_size
        i = torch.randint(0, H - ph, (1,)).item()
        j = torch.randint(0, W - pw, (1,)).item()
        return img[i:i+ph, j:j+pw]

def get_valid_sequences(sequence_paths, out_file="sequences_left_out.txt"):
    """
    Filtra las secuencias que no cumplen con los requisitos:
    - Debe existir pre-processing.txt
    - El valor 'a' debe estar cerca de 1 (abs(a-1) <= 0.2)
    - Debe haber archivos .tif
    - Al menos uno de los canales (o la secuencia completa) debe tener >= 5 frames
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
                a, b = np.loadtxt(preproc_file)
            except Exception as e:
                f_out.write(f"{seq.name}, error reading pre-processing.txt: {e}\n")
                continue

            if np.abs(a-1) > 0.2:
                f_out.write(f"{seq.name}, a={a}\n")
                continue
            
            tif_files = sorted(seq.glob("*.tif"))
            if not tif_files:
                continue
            
            # Verificar si hay suficientes frames en al menos un canal
            names = [f.name for f in tif_files]
            channels = []
            if any("_c0_" in n for n in names):
                channels.append("_c0_")
            if any("_c1_" in n for n in names):
                channels.append("_c1_")
            if not channels:
                channels = [""]
            
            has_enough_frames = False
            for ch in channels:
                if ch:
                    frames = [f for f in tif_files if ch in f.name]
                else:
                    frames = tif_files
                if len(frames) >= 5:
                    has_enough_frames = True
                    break
            
            if has_enough_frames:
                valid_sequences.append((str(seq), float(a), float(b)))
            else:
                f_out.write(f"{seq.name}, not enough frames (min 5)\n")
        valid_sequences = sorted(valid_sequences)
    return valid_sequences

class FastDVDnetDataset(Dataset):
    def __init__(self, sequence_info, patch_size=None, transform=None, data_scale=255.0):
        """
        Dataset para FastDVDnet:
        - Soporta imágenes .tif
        - Aplica transformación lineal (a,b) desde pre_processing.txt
        - Genera stacks de 5 frames
        - Crop aleatorio opcional
        
        sequence_info: lista de tuples (path, a, b)
        """
        self.patch_size = patch_size
        self.transform = transform
        self.data_scale = data_scale
        self.stacks = []

        for seq_path, a, b in sequence_info:
            seq = Path(seq_path)
            tif_files = sorted(seq.glob("*.tif"))
            names = [f.name for f in tif_files]
            channels = []
            if any("_c0_" in n for n in names):
                channels.append("_c0_")
            if any("_c1_" in n for n in names):
                channels.append("_c1_")
            if not channels:
                channels = [""]
            
            for ch in channels:
                if ch:
                    frames = sorted(f for f in tif_files if ch in f.name)
                else:
                    frames = sorted(tif_files)
                
                if len(frames) < 5:
                    continue
                
                for i in range(2, len(frames) - 2):
                    stack_paths = [
                        frames[i - 2],
                        frames[i - 1],
                        frames[i],
                        frames[i + 1],
                        frames[i + 2],
                    ]
                    self.stacks.append((stack_paths, float(a), float(b)))

    def __len__(self): #I can change that to have more iterations per epoch
        return len(self.stacks)

    def make_divisible_by_4(self, img):
        H, W = img.shape[-2], img.shape[-1]
        H4 = (H // 4) * 4
        W4 = (W // 4) * 4
        return img[..., :H4, :W4]

    def _read_tif(self, path):
        """Lee un .tif y lo convierte a tensor [1,H,W] sin normalizar (crudo)"""
        img = iio.imread(str(path))
        img = torch.from_numpy(img).float()

        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3:
            img = img.permute(2, 0, 1)
            img = img.mean(dim=0, keepdim=True)

        return img

    def __getitem__(self, idx):
        stack_paths, a, b = self.stacks[idx]

        central_frame = self._read_tif(stack_paths[2])
        frames = []
        for p in stack_paths:
            if p is None:
                frames.append(torch.zeros_like(central_frame))
            elif p == stack_paths[2]:
                frames.append(central_frame)
            else:
                frames.append(self._read_tif(p))

        stack = torch.cat(frames, dim=0)  # [5,H,W]
        stack = self.make_divisible_by_4(stack)
        
        # Valery's linear transform / data_scale
        stack = linear_transform(stack, a, b, u=1) / self.data_scale # u=1 para los nuevos pesos que me pasó Valéry
        stack = torch.clamp(stack, min=0.0)

        # Random crop
        if self.patch_size is not None:
            H, W = stack.shape[1:]
            ph, pw = self.patch_size
            if H >= ph and W >= pw:
                top = torch.randint(0, H - ph + 1, (1,)).item()
                left = torch.randint(0, W - pw + 1, (1,)).item()
                stack = stack[:, top:top + ph, left:left + pw]
        if self.transform:
            stack = self.transform(stack)
        
        target = stack[2:3, :, :].clone()  # Central frame [1,H,W]

        return stack, target  # [5,H,W], [1,H,W]
def get_fmdd_sequences(root_dir, modalities=None):
    """
    Descubre secuencias en la estructura de FMDD:
    root_dir/Modality/raw/SequenceID/*.png
    """
    root = Path(root_dir)
    sequences = []
    
    # Si no se especifican modalidades, buscar en todas
    if modalities is None:
        modalities = sorted([d.name for d in root.iterdir() if d.is_dir()])
        modalities = [m for m in modalities if (root / m / "raw").exists()]
    
    for mod in modalities:
        mod_raw = root / mod / "raw"
        if not mod_raw.exists():
            continue
        
        # Sorteamos para que el orden sea determinista
        for seq_dir in sorted(mod_raw.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            png_files = sorted(seq_dir.glob("*.png"))
            if len(png_files) >= 5:
                # Buscar ground truth correspondiente
                gt_path = root / mod / "gt" / seq_dir.name / "avg50.png"
                if not gt_path.exists():
                    gt_path = None
                
                sequences.append({
                    'modality': mod,
                    'seq_id': seq_dir.name,
                    'frames': [str(p) for p in png_files],
                    'gt': str(gt_path) if gt_path else None
                })
    
    return sequences

def get_fmdd_split_from_file(sequences, split_file):
    """
    Carga la partición desde un archivo TXT.
    Retorna (train_seqs, test_seqs, visualize_indices)
    """
    split_info = {}
    viz_indices = []
    
    with open(split_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                split_info[parts[0]] = parts[1] # "mod/seq" -> "train" or "test"
            if len(parts) >= 3 and parts[2].lower() == "true":
                # Guardaremos el path para identificarlo luego en la lista de test
                viz_indices.append(parts[0])

    train_seqs = []
    test_seqs = []
    
    for s in sequences:
        key = f"{s['modality']}/{s['seq_id']}"
        role = split_info.get(key, "train")
        if role in ["test", "val"]: # Indistintamente
            test_seqs.append(s)
        else:
            train_seqs.append(s)
            
    # Mapear viz_indices a los índices reales en la lista test_seqs
    final_viz_indices = []
    for i, s in enumerate(test_seqs):
        key = f"{s['modality']}/{s['seq_id']}"
        if key in viz_indices:
            final_viz_indices.append(i)
            
    return train_seqs, test_seqs, final_viz_indices

class FMDDataset(Dataset):
    def __init__(self, sequence_info, patch_size=None, transform=None, data_scale=255.0, a=1.0, b=0.0, mode='raw', gamma=None, num_frames=5):
        """
        Dataset para FMDD:
        - Soporta imágenes .png
        - mode:
            'raw'       → usa imágenes reales ruidosas del dataset
            'synthetic' → usa GT + ruido Poisson generado internamente por el dataset
            'clean'     → devuelve solo la GT limpia (sin ruido); el ruido lo aplica physics externamente
        - gamma: Ganancia para el ruido de Poisson sintético (solo en mode='synthetic')
        - num_frames: Número de frames en el stack (5 para FastDVDnet, 1 para DRUNet)
        """
        self.patch_size = patch_size
        self.transform = transform
        self.data_scale = data_scale
        self.a = a
        self.b = b
        self.mode = mode
        self.gamma = gamma
        self.num_frames = num_frames
        self.stacks = []

        for seq in sequence_info:
            frames = seq['frames']
            gt = seq['gt']
            
            if self.mode == 'raw':
                if len(frames) >= self.num_frames:
                    for i in range(len(frames) - (self.num_frames - 1)):
                        stack_paths = frames[i:i+self.num_frames]
                        self.stacks.append((stack_paths, gt))
            elif self.mode == 'synthetic':
                if gt:
                    # En modo sintético, cada "stack" se genera desde la misma GT
                    # Podemos añadir múltiples entradas por secuencia para aumentar el dataset
                    # por ejemplo, 55 stacks por imagen GT para llegar a ~12000 parches
                    for _ in range(55):
                        self.stacks.append((None, gt))
            elif self.mode == 'clean':
                if gt:
                    # En modo clean solo devolvemos el GT. El ruido lo aplica physics externamente.
                    # Múltiples entradas por GT para aumentar el dataset (como en synthetic).
                    for _ in range(55):
                        self.stacks.append((None, gt))

    def __len__(self):
        return len(self.stacks)

    def make_divisible_by_4(self, img):
        H, W = img.shape[-2], img.shape[-1]
        H4 = (H // 4) * 4
        W4 = (W // 4) * 4
        return img[..., :H4, :W4]

    def _read_png(self, path):
        """Lee un .png y lo convierte a tensor [1,H,W] normalizado"""
        img = iio.imread(str(path))
        img = torch.from_numpy(img).float()
        img = img / 255.0  # Normalize to [0, 1]

        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3:
            # Si es RGB, convertir a gris promediando
            img = img.permute(2, 0, 1)
            img = img.mean(dim=0, keepdim=True)

        return img

    def _add_poisson_noise(self, img):
        """Añade ruido de Poisson sintético [0, 1]"""
        if self.gamma is None:
            return img
        # Poisson(img * gamma) / gamma
        # Aseguramos que img sea no negativa
        img = torch.clamp(img, min=0.0)
        noisy = torch.poisson(img * self.gamma) / self.gamma
        return noisy

    def __getitem__(self, idx):
        stack_paths, gt_path = self.stacks[idx]

        if self.mode == 'raw':
            # Cargar frames ruidosos
            frames = [self._read_png(p) for p in stack_paths]
            stack = torch.cat(frames, dim=0)  # [num_frames,H,W]
            
            # Cargar GT si existe
            if gt_path:
                target_full = self._read_png(gt_path)
            else:
                # Si no hay GT, el target es el frame central
                mid = self.num_frames // 2
                target_full = stack[mid:mid+1, :, :].clone()

            # Random crop sobre ambos simultáneamente
            if self.patch_size is not None:
                H, W = stack.shape[1:]
                ph, pw = self.patch_size
                if H >= ph and W >= pw:
                    top = torch.randint(0, H - ph + 1, (1,)).item()
                    left = torch.randint(0, W - pw + 1, (1,)).item()
                    stack = stack[:, top:top + ph, left:left + pw]
                    target = target_full[:, top:top + ph, left:left + pw]
                else:
                    target = target_full
            else:
                target = target_full

        elif self.mode == 'clean':
            # Devuelve solo la imagen GT limpia. El ruido lo aplica physics externamente.
            clean_full = self._read_png(gt_path)

            if self.patch_size is not None:
                H, W = clean_full.shape[1:]
                ph, pw = self.patch_size
                if H >= ph and W >= pw:
                    top = torch.randint(0, H - ph + 1, (1,)).item()
                    left = torch.randint(0, W - pw + 1, (1,)).item()
                    clean_patch = clean_full[:, top:top + ph, left:left + pw]
                else:
                    clean_patch = clean_full
            else:
                clean_patch = clean_full

            # stack y target son ambos la imagen limpia; el training loop añadirá el ruido
            stack = clean_patch.expand(self.num_frames, -1, -1).clone()  # [num_frames,H,W]
            target = clean_patch

        else: # synthetic
            # Cargar la imagen GT una sola vez
            clean_full = self._read_png(gt_path)
            
            # Crop PRIMERO sobre la imagen limpia para ahorrar CPU
            if self.patch_size is not None:
                H, W = clean_full.shape[1:]
                ph, pw = self.patch_size
                if H >= ph and W >= pw:
                    top = torch.randint(0, H - ph + 1, (1,)).item()
                    left = torch.randint(0, W - pw + 1, (1,)).item()
                    clean_patch = clean_full[:, top:top + ph, left:left + pw]
                else:
                    clean_patch = clean_full
            else:
                clean_patch = clean_full

            # Añadir ruido SOLO al fragmento recortado (Mucha menos CPU!)
            frames = [self._add_poisson_noise(clean_patch) for _ in range(self.num_frames)]
            stack = torch.cat(frames, dim=0)
            target = clean_patch

        # Normalización común
        stack = self.make_divisible_by_4(stack)
        stack = linear_transform(stack, self.a, self.b, u=1) / self.data_scale
        stack = torch.clamp(stack, min=0.0)
        #print(stack.min(), stack.max())

        target = self.make_divisible_by_4(target)
        target = linear_transform(target, self.a, self.b, u=1) / self.data_scale
        target = torch.clamp(target, min=0.0)

        # Aplicar transformación (ej: D4) a AMBOS para mantener consistencia
        if self.transform:
            combined = torch.cat([stack, target], dim=0)
            combined = self.transform(combined)
            stack = combined[:self.num_frames, :, :]
            target = combined[self.num_frames:, :, :]

        return stack, target
