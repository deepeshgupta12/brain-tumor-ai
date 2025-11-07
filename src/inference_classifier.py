import os, torch, torch.nn as nn
import numpy as np

class TinyCNN(nn.Module):
    """Minimal CNN for smoke tests; replace with timm/ResNet later."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.act(self.fc(h))

def load_classifier(weights_path: str | None = None, device: str = "cpu") -> TinyCNN:
    model = TinyCNN().to(device)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

@torch.inference_mode()
def predict_has_tumor(model: TinyCNN, arr3c: np.ndarray, device: str = "cpu") -> float:
    x = torch.from_numpy(arr3c).unsqueeze(0).to(device)  # (1,3,224,224)
    p = model(x).item()
    return float(p)
