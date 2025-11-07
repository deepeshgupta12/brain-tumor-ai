import os
import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query

from app.schemas import PredictResponse, MetricsRequest, MetricsResponse
from src.io_utils import load_image_file, load_dicom_zip_to_volume
from src.inference_timm import TimmClassifier
from src.explain import GradCAM, overlay_cam

CLS_META = "models/metadata.json"
CLS_W    = "models/classifier_state.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

_classifier = None  # type: TimmClassifier
def ensure_classifier():
    global _classifier
    if _classifier is None:
        _classifier = TimmClassifier(meta_path=CLS_META, ckpt_path=CLS_W, device=DEVICE)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_classifier()
    yield

app = FastAPI(title="Brain Tumor Classifier (Not for Diagnostic Use)", lifespan=lifespan)

def _prepare_input_from_png(upload) -> np.ndarray:
    return load_image_file(upload)  # (H,W) float [0,1]

def _prepare_input_from_dicom_zip(data: bytes) -> np.ndarray:
    vol = load_dicom_zip_to_volume(data)  # (Z,H,W)
    return vol[vol.shape[0] // 2]         # center slice

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    save_cam: bool = Query(False, description="Save Grad-CAM overlay to /tmp/last_cam.png"),
    tta: bool = Query(False, description="Use test-time augmentation (slower, can improve accuracy)")
):
    ensure_classifier()
    fname = (file.filename or "").lower()
    try:
        if fname.endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
            arr_hw = _prepare_input_from_png(file.file)
        elif fname.endswith(".zip"):
            data = await file.read()
            arr_hw = _prepare_input_from_dicom_zip(data)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type. Upload PNG/JPG/TIF or a zipped DICOM series.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read file: {e}")

    # Classify (with optional TTA)
    probs = _classifier.predict_proba(arr_hw, tta=tta)
    class_names = _classifier.class_names
    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]
    tumor_prob = _classifier.tumor_probability(probs)

    THRESH = float(os.getenv("TUMOR_THRESH", "0.22"))
    has_tumor = bool(tumor_prob >= THRESH)
    notes = f"Triage only. Not for diagnostic use. WHO CNS5 requires integrated diagnosis. TTA={'on' if tta else 'off'}."

    if save_cam:
        from PIL import Image
        from torchvision import transforms as T
        img_size = _classifier.img_size
        tf = T.Compose([T.Resize((img_size,img_size)), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
        arr8 = (np.clip(arr_hw,0,1)*255).astype("uint8")
        pil = Image.fromarray(arr8, mode="L").convert("RGB")
        x = tf(pil).unsqueeze(0).to(_classifier.device)
        cam = GradCAM(_classifier.model, device=_classifier.device).generate(x, target_index=pred_idx)
        bg = np.array(pil.resize((img_size,img_size)).convert("L"), dtype=np.float32)/255.0
        overlay = overlay_cam(bg, cam, alpha=0.35)
        out_path = "/tmp/last_cam.png"
        import cv2
        cv2.imwrite(out_path, overlay)
        notes += f" | Grad-CAM saved: {out_path}"

    return PredictResponse(
        has_tumor=has_tumor,
        tumor_probability=round(float(tumor_prob),4),
        classification=pred_name,
        segmentation_available=False,
        tumor_volume_ml=None,
        notes=notes
    )

# ---------- /metrics (with TTA) ----------
ALIASES = {
    "no_tumor":"no_tumor","no-tumor":"no_tumor","notumor":"no_tumor",
    "glioma_tumor":"glioma","glioma":"glioma",
    "meningioma_tumor":"meningioma","meningioma_tomor":"meningioma","meningioma":"meningioma",
    "pituitary_tumor":"pituitary","pituitary":"pituitary",
}
CLASS_TO_IDX = {"no_tumor":0,"glioma":1,"meningioma":2,"pituitary":3}

def _infer_label_from_path(p: str):
    import os
    parent = os.path.basename(os.path.dirname(p)).strip().lower().replace(" ","_")
    cname = ALIASES.get(parent, parent)
    return CLASS_TO_IDX.get(cname, None)

def _threshold_sweep(y_true, y_score):
    import numpy as np
    ts = np.linspace(0.0, 1.0, 101)
    best_t, best_j = 0.5, -1.0
    for t in ts:
        y_pred = (y_score >= t).astype(int)
        tp = int(((y_pred==1) & (y_true==1)).sum())
        tn = int(((y_pred==0) & (y_true==0)).sum())
        fp = int(((y_pred==1) & (y_true==0)).sum())
        fn = int(((y_pred==0) & (y_true==1)).sum())
        tpr = tp / max(tp+fn, 1)
        fpr = fp / max(fp+tn, 1)
        J = tpr - fpr
        if J > best_j:
            best_j, best_t = J, t
    return float(best_t), float(best_j)

@app.post("/metrics", response_model=MetricsResponse)
async def metrics(req: MetricsRequest):
    ensure_classifier()
    import os
    import glob
    import time
    from PIL import Image
    from sklearn.metrics import confusion_matrix

    # Gather paths
    paths = []
    if req.paths: 
        paths.extend(req.paths)
    if req.root:
        exts = ("png","jpg","jpeg","bmp","tif","tiff")
        if req.recursive:
            for ext in exts:
                paths.extend(glob.glob(os.path.join(req.root, "**", f"*.{ext}"), recursive=True))
        else:
            for ext in exts:
                paths.extend(glob.glob(os.path.join(req.root, f"*.{ext}")))
    paths = sorted(set(paths))
    n_total = len(paths)
    if not paths:
        raise HTTPException(status_code=400, detail="No images found for evaluation.")
    if req.img_limit and n_total > req.img_limit:
        paths = paths[:req.img_limit]

    # Run inference
    preds_mc, tumor_probs, y_true_mc = [], [], []
    t0 = time.time()
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            arr = np.asarray(img).astype(np.float32)/255.0
            pr = _classifier.predict_proba(arr, tta=req.tta)
            preds_mc.append(int(np.argmax(pr)))
            tumor_probs.append(float(1.0 - pr[0]))
            y_true_mc.append(_infer_label_from_path(p))
        except Exception:
            continue
    dt = time.time() - t0

    labeled_idx = [i for i,g in enumerate(y_true_mc) if g is not None]
    n_scored = len(preds_mc); 
    n_labeled = len(labeled_idx)

    acc_mc = acc_bin = None
    cm_mc = cm_bin = None
    thr = float(os.getenv("TUMOR_THRESH", str(req.threshold if req.threshold is not None else 0.22)))
    best_t = best_j = None

    if n_labeled > 0:
        y_mc = np.array([y_true_mc[i] for i in labeled_idx], dtype=int)
        p_mc = np.array([preds_mc[i]   for i in labeled_idx], dtype=int)
        acc_mc = float((y_mc == p_mc).mean())
        cm_mc = confusion_matrix(y_mc, p_mc, labels=[0,1,2,3]).tolist()

        y_bin = (y_mc != 0).astype(int)
        tprob = np.array([tumor_probs[i] for i in labeled_idx], dtype=float)
        pred_bin = (tprob >= thr).astype(int)
        acc_bin = float((pred_bin == y_bin).mean())
        cm_bin = confusion_matrix(y_bin, pred_bin, labels=[0,1]).tolist()
        if (y_bin==0).any() and (y_bin==1).any():
            best_t, best_j = _threshold_sweep(y_bin, tprob)

    samples = None
    if req.return_predictions:
        CLASS_NAMES = _classifier.class_names
        k = min(req.max_predictions, n_scored)
        samples = [{
            "path": paths[i],
            "pred_class": CLASS_NAMES[preds_mc[i]],
            "tumor_prob": round(tumor_probs[i],4),
            "true_class": None if y_true_mc[i] is None else CLASS_NAMES[y_true_mc[i]],
        } for i in range(k)]

    return MetricsResponse(
        n_total=n_total, n_scored=n_scored, n_labeled=n_labeled,
        acc_multiclass=acc_mc, acc_binary=acc_bin,
        cm_multiclass=cm_mc, cm_binary=cm_bin,
        suggested_threshold_by_YoudenJ=best_t, best_J=best_j,
        notes=f"Processed in {dt:.1f}s on {_classifier.device}. TTA={'on' if req.tta else 'off'}. Labels inferred from parent folders.",
        samples=samples
    )
