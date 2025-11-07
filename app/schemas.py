from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class PredictResponse(BaseModel):
    has_tumor: bool
    tumor_probability: float
    classification: str
    segmentation_available: bool = False
    tumor_volume_ml: Optional[float] = None
    notes: str = "Triage only. Not for diagnostic use."

class MetricsRequest(BaseModel):
    root: Optional[str] = Field(None, description="Folder to scan for images")
    paths: Optional[List[str]] = Field(None, description="Explicit image file paths (PNG/JPG/TIF)")
    recursive: bool = True
    img_limit: int = 0
    threshold: Optional[float] = None
    return_predictions: bool = False
    max_predictions: int = 50
    tta: bool = False  # <---- NEW

class MetricsResponse(BaseModel):
    n_total: int
    n_scored: int
    n_labeled: int
    acc_multiclass: Optional[float] = None
    acc_binary: Optional[float] = None
    cm_multiclass: Optional[List[List[int]]] = None
    cm_binary: Optional[List[List[int]]] = None
    suggested_threshold_by_YoudenJ: Optional[float] = None
    best_J: Optional[float] = None
    notes: str
    samples: Optional[List[Dict]] = None
