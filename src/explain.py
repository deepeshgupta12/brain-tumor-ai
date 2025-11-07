import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

def _find_last_conv(module: nn.Module) -> Optional[nn.Module]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None, device: str = "cpu"):
        self.model = model
        self.model.eval()
        self.device = device
        self.target = target_layer or _find_last_conv(model)
        if self.target is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM.")
        self._acts = None
        self._grads = None
        # hooks
        self.target.register_forward_hook(self._hook_acts)
        self.target.register_full_backward_hook(self._hook_grads)

    def _hook_acts(self, module, inp, out):
        self._acts = out.detach()

    def _hook_grads(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def generate(self, x: torch.Tensor, target_index: Optional[int] = None) -> np.ndarray:
        """
        x: (1,3,H,W) tensor; model returns logits (no softmax).
        target_index: int class index to explain. If None, uses argmax.
        returns: (H',W') cam in [0,1], resized to 224x224 (or HxW if H==W==224).
        """
        x = x.to(self.device).requires_grad_(True)
        out = self.model(x)                 # (1,C)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if target_index is None:
            target_index = int(out.argmax(dim=1).item())
        score = out[0, target_index]        # <-- scalar logit
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)  # d(score)/d(features)

        acts = self._acts                   # (1,C,h,w)
        grads = self._grads                 # (1,C,h,w)
        weights = grads.mean(dim=(2,3), keepdim=True)     # (1,C,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=False)  # (1,h,w)
        cam = torch.relu(cam)[0].cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-6)
        # Standardize output size at 224x224 for overlay convenience
        cam = cv2.resize(cam, (224,224), interpolation=cv2.INTER_LINEAR)
        return cam

def overlay_cam(gray224: np.ndarray, cam224: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    gray224: (224,224) float32 [0..1] background image
    cam224:  (224,224) float32 [0..1] heatmap
    returns BGR uint8 overlay (224,224,3)
    """
    base = (np.clip(gray224, 0, 1) * 255).astype(np.uint8)
    heat = (np.clip(cam224, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base3 = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    over = cv2.addWeighted(heat, alpha, base3, 1 - alpha, 0)
    over = cv2.cvtColor(over, cv2.COLOR_RGB2BGR)
    return over
