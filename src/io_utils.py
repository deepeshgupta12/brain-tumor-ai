import os, zipfile, tempfile, io
import numpy as np, cv2, pydicom

def load_image_file(file_like) -> np.ndarray:
    """Read a single image file (png/jpg/jpeg/tif) to float32 [0,1], HxW."""
    data = file_like.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Unsupported or corrupt image.")
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0)

def load_dicom_zip_to_volume(upload_bytes: bytes) -> np.ndarray:
    """Accepts bytes of a .zip containing a DICOM series; returns volume (Z,H,W) in [0,1]."""
    tmpdir = tempfile.mkdtemp(prefix="dicom_")
    zpath = os.path.join(tmpdir, "series.zip")
    with open(zpath, "wb") as f:
        f.write(upload_bytes)
    with zipfile.ZipFile(zpath, 'r') as zf:
        zf.extractall(tmpdir)
    dsets = []
    for root, _, files in os.walk(tmpdir):
        for name in files:
            fpath = os.path.join(root, name)
            try:
                d = pydicom.dcmread(fpath, force=True)
                if hasattr(d, "pixel_array"):
                    dsets.append(d)
            except Exception:
                pass
    if not dsets:
        raise ValueError("No readable DICOM images found in zip.")
    dsets.sort(key=lambda d: getattr(d, "InstanceNumber", 0))
    vol = [d.pixel_array.astype(np.float32) for d in dsets]
    vol = np.stack(vol, axis=0)  # (Z,H,W)
    vmin, vmax = vol.min(), vol.max()
    vol = (vol - vmin) / (vmax - vmin + 1e-6)
    return vol
