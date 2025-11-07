import json, os
from pathlib import Path
from typing import Optional, List, Callable

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import timm
from torchvision import transforms as T
import torchvision.transforms.functional as TF

DEFAULT_MEAN = [0.5, 0.5, 0.5]
DEFAULT_STD  = [0.5, 0.5, 0.5]

def _disable_inplace_relu(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

class TimmClassifier:
    def __init__(self, meta_path="models/metadata.json", ckpt_path="models/classifier_state.pt", device=None):
        meta_path = Path(meta_path); ckpt_path = Path(ckpt_path)
        if not (meta_path.exists() and ckpt_path.exists()):
            raise FileNotFoundError(f"Missing model files: {meta_path} / {ckpt_path}")

        meta = json.loads(Path(meta_path).read_text())
        self.arch = meta.get("arch", "resnet18")
        self.img_size = int(meta.get("img_size", 224))
        self.class_names = meta.get("class_names", ["no_tumor","glioma","meningioma","pituitary"])

        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = timm.create_model(self.arch, pretrained=False, num_classes=len(self.class_names), in_chans=3)
        _disable_inplace_relu(self.model)  # safer for Grad-CAM
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
        self.model.eval().to(self.device)

        self.tf = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ])

    def _to_pil_rgb(self, arr: np.ndarray) -> Image.Image:
        """arr in [0,1] (H,W) or (H,W,3) -> PIL RGB"""
        if arr.ndim == 2:
            arr = (np.clip(arr, 0, 1) * 255).astype("uint8")
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = (np.clip(arr, 0, 1) * 255).astype("uint8")
            return Image.fromarray(arr, mode="RGB")
        raise ValueError("Expected 2D grayscale or HWC RGB array in [0,1].")

    @torch.inference_mode()
    def _forward_pil(self, pil: Image.Image) -> np.ndarray:
        x = self.tf(pil).unsqueeze(0).to(self.device)  # (1,3,H,W)
        logits = self.model(x)
        return torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    def _tta_set(self) -> List[Callable[[Image.Image], Image.Image]]:
        # 6-way TTA: id, hflip, vflip, rot90, rot180, rot270
        return [
            lambda im: im,
            lambda im: TF.hflip(im),
            lambda im: TF.vflip(im),
            lambda im: TF.rotate(im, 90),
            lambda im: TF.rotate(im, 180),
            lambda im: TF.rotate(im, 270),
        ]

    @torch.inference_mode()
    def predict_proba(self, arr_hw_or_hwc: np.ndarray, tta: bool = False) -> np.ndarray:
        pil = self._to_pil_rgb(arr_hw_or_hwc)
        if not tta:
            return self._forward_pil(pil)

        # Average probabilities across TTA members
        probs = None
        for aug in self._tta_set():
            p = self._forward_pil(aug(pil))
            probs = p if probs is None else (probs + p)
        probs /= float(len(self._tta_set()))
        return probs

    def tumor_probability(self, probs: np.ndarray) -> float:
        # class 0 = no_tumor by our training; tumor prob = 1 - P(no_tumor)
        return float(1.0 - float(probs[0]))
