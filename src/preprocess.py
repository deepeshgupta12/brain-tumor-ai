import numpy as np

def center_crop2d(img: np.ndarray, size: int = 224):
    h, w = img.shape[-2:]
    y0 = max((h - size) // 2, 0)
    x0 = max((w - size) // 2, 0)
    return img[..., y0:y0+size, x0:x0+size]

def zscore(x: np.ndarray):
    m = x.mean()
    s = x.std() + 1e-6
    return (x - m) / s

def make_3slice_stack(vol: np.ndarray):
    """vol: (Z,H,W) in [0,1] -> (3,224,224) as pseudo-RGB stack using middle ±k slices."""
    if vol.ndim == 2:  # single image fallback
        vol = np.stack([vol]*9, axis=0)
    z = vol.shape[0] // 2
    k = max(1, vol.shape[0] // 12)
    sl = [vol[max(z-k,0)], vol[z], vol[min(z+k, vol.shape[0]-1)]]
    arr = np.stack(sl, axis=0)  # (3,H,W)
    return center_crop2d(arr, 224).astype("float32")
