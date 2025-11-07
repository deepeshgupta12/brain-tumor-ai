import os, numpy as np, torch
from typing import Tuple

def load_seg_model(weights_path: str | None = None):
    # Placeholder; wire a MONAI UNet later. Return None if no weights.
    if weights_path and os.path.exists(weights_path):
        # lazy import to keep deps light at boot
        from monai.networks.nets import UNet
        net = UNet(spatial_dims=3, in_channels=1, out_channels=1,
                   channels=(16,32,64,128), strides=(2,2,2))
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        net.eval()
        return net
    return None

@torch.inference_mode()
def seg_and_volume(net, vol3d: np.ndarray, voxel_mm=(1.0,1.0,1.0)) -> Tuple[np.ndarray, float]:
    """vol3d: (Z,H,W) in [0,1]. Returns (mask, volume_ml)."""
    if net is None:
        return np.zeros_like(vol3d, dtype=np.uint8), 0.0
    x = torch.from_numpy(vol3d[None,None]).float()  # (1,1,Z,H,W)
    logits = net(x)
    mask = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0,0].astype(np.uint8)
    vox_ml = (voxel_mm[0]*voxel_mm[1]*voxel_mm[2]) / 1000.0
    return mask, float(mask.sum() * vox_ml)
