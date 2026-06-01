import os
import numpy as np
import torch
import random
import fire

def generar_lista_tif(directorio, archivo_salida="lista_tif.txt"):
    lista_tif = []

    for root, dirs, files in os.walk(directorio, followlinks=True):
        # evitar entrar en directorios llamados "check"
        dirs[:] = [d for d in dirs if d != "check"]

        for file in files:
            if file.lower().endswith(".tif"):
                lista_tif.append(os.path.join(root, file))

    lista_tif.sort()
    with open(archivo_salida, "w") as f:
        for ruta in lista_tif:
            f.write(ruta + "\n")

    print(f"Se encontraron {len(lista_tif)} archivos .tif")

def resample_poisson_sequence(y, gamma, gamma_target=1.4, seed=None):
    """
    Normaliza una secuencia de imágenes Poisson a una ganancia target usando remuestreo.

    Args:
        y (np.ndarray): Secuencia original (imagen o stack de imágenes) con ruido Poisson escalado.
        gamma (float): Ganancia original de la secuencia.
        gamma_target (float): Ganancia target a la que queremos normalizar.
        seed (int, opcional): Semilla para reproducibilidad del remuestreo.

    Returns:
        np.ndarray: Secuencia remuestreada con ganancia gamma_target.
    """
    if seed is not None:
        np.random.seed(seed)

    # Paso 1: Poisson base (aproximadamente)
    k_base = y / gamma  # Esto es k_i ≈ P(x / gamma)
    # Paso 2: Re-muestreo Poisson para ganancia target. La media de la nueva Poisson es k_base * (gamma / gamma_target)
    k_target = np.random.poisson(lam=k_base * (gamma / gamma_target))
    # Paso 3: Escalar al gamma_target
    y_norm = gamma_target * k_target

    return torch.from_numpy(y_norm).to(torch.float32)

def linear_transform(data, a, b, u=1.4, v=0, inverse=False):
    alpha =  u/a
    beta  = v - u*b/a

    if inverse:
        return (data-beta) / alpha
    return alpha*data+beta

class RandomD4:
    """
    Aplica una transformación del grupo D4 (rotaciones 90 + flips)
    Funciona para [H,W], [C,H,W] o [T,H,W]
    """
    def __call__(self, x):
        # Rotación
        k = random.randint(0, 3)
        x = torch.rot90(x, k, dims=[-2, -1])

        # Flip horizontal
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-1])

        return x

if __name__ == "__main__":
    fire.Fire()