# Brain Tumor Classifier (Research/Triage Only)

FastAPI service + PyTorch/timm CNN that classifies brain MRI images into **four classes**:
**`no_tumor`**, **`glioma`**, **`meningioma`**, **`pituitary`**.

Includes:
- Grad-CAM heatmaps for explainability
- `/metrics` endpoint for batch evaluation (folder or explicit paths)
- Dataset-from-folders → CSV builder
- Evaluation scripts (confusion matrices, ROC/PR, threshold sweep)
- Test-Time Augmentation (TTA) at inference
- Apple Silicon–friendly training recipe

> **Safety disclaimer**  
> This software is **not a medical device** and must **not** be used for diagnosis or patient management. Research/education only.

---

## Features
- 🧠 4-class classification with **binary tumor probability** (`tumor_prob = 1 - P(no_tumor)`).
- 🔥 **Grad-CAM** overlays (`?save_cam=true`) to visualize salient regions.
- 📊 **/metrics** endpoint: accuracy on a folder or list of files; optional **TTA**.
- 🧰 Dataset builder from class-named folders → `train.csv` / `val.csv`.
- 📈 Evaluation with confusion matrices, ROC/PR, threshold sweep, and `predictions.csv`.
- ⚙️ Training scripts: simple (resnet18) and stronger (resnet34/50 + augments, label smoothing, optional mixup/cutmix).
- 🍎 Tips for Apple Silicon (M1/M2) resource use.

---

## Project Structure (key files)
```
app/
  main.py          # FastAPI app (/predict, /metrics)
  schemas.py       # Pydantic request/response models
models/            # (gitignored weights) classifier_state.pt, metadata.json
scripts/
  build_from_folders.py  # dataset → CSVs
  train_classifier.py    # simple trainer (resnet18)
  train_classifier_v2.py # stronger trainer (resnet34/50, augs, label smoothing, mixup)
  eval_classifier.py     # evaluation + plots + threshold sweep
src/
  inference_timm.py # model loader + predict_proba (with optional TTA)
  io_utils.py       # image & DICOM helpers
  explain.py        # Grad-CAM implementation
tests/
  test_api.py       # minimal endpoint test (FastAPI TestClient)
requirements.txt
README.md
LICENSE
```

---

## Requirements
- Python 3.10+
- macOS / Linux / Windows
- (Optional) Apple Silicon: PyTorch MPS backend

### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset → CSV (from class-named folders)
Expected folder names (case/underscore tolerant; a few aliases supported):
```
data/images/train/{no_tumor,glioma,meningioma,pituitary}
data/images/val/{no_tumor,glioma,meningioma,pituitary}
```

Generate CSVs from a **single** root (auto-split) or from separate roots:

```bash
# Example: from a single root, 80/20 split
python scripts/build_from_folders.py   --root data/images/train   --out  data/folder_csv   --val_frac 0.2   --seed 42
```

This creates `data/folder_csv/train.csv` and `data/folder_csv/val.csv` with columns:
`path, class_name, multiclass_label, binary_label`.

---

## Training

### Simple (resnet18 @ 224px)
```bash
python scripts/train_classifier.py   --train_csv data/folder_csv/train.csv   --val_csv   data/folder_csv/val.csv   --out_dir   runs/cls_resnet18   --arch      resnet18   --epochs    10   --batch_size 32   --img_size  224   --lr        3e-4
```

### Stronger (resnet34 @ 224px, label smoothing; good on M1)
```bash
python scripts/train_classifier_v2.py   --train_csv data/folder_csv/train.csv   --val_csv   data/folder_csv/val.csv   --out_dir   runs/cls_rn34_224   --arch      resnet34   --epochs    12   --batch_size 12   --img_size  224   --lr 5e-4 --weight_decay 0.05   --label_smoothing 0.1   --use_mixup false   --num_workers 0
```

> The trainer copies the **best checkpoint** to `models/classifier_state.pt` and writes `models/metadata.json`. The API reads from there automatically.

**Apple Silicon tips**
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OMP_NUM_THREADS=1
# and prefer --num_workers 0 for dataloaders
```

---

## Evaluation
```bash
python scripts/eval_classifier.py   --csv data/folder_csv/val.csv   --out runs/eval_model_val   --meta models/metadata.json   --ckpt models/classifier_state.pt   --threshold 0.10
```

Artifacts (in `runs/eval_model_val/`):
- `cm_multiclass.png`, `cm_binary.png`
- `roc_binary.png`, `pr_binary.png`
- `threshold_sweep.png`
- `predictions.csv`, `eval_report.json`

> The script prints a **suggested tumor threshold** (Youden-J). This may vary by model/dataset; we’ve found **0.10** a strong default for RN34@224.

---

## API Quickstart
Run the service with your threshold:
```bash
TUMOR_THRESH=0.10 uvicorn app.main:app --reload --port 8000
```
Docs: http://127.0.0.1:8000/docs

### `/predict` (multipart)
- **file**: PNG/JPG/TIF or zipped **DICOM** series  
- **save_cam** (bool): save Grad-CAM overlay to `/tmp/last_cam.png`  
- **tta** (bool): enable Test-Time Augmentation

Example:
```bash
curl -X POST "http://127.0.0.1:8000/predict?save_cam=true&tta=true"   -H "accept: application/json" -H "Content-Type: multipart/form-data"   -F "file=@/path/to/image.jpg;type=image/jpeg"
```

### `/metrics` (JSON)
Evaluate a folder (labels inferred from parent folders) or explicit file paths.
```bash
curl -X POST "http://127.0.0.1:8000/metrics"   -H "Content-Type: application/json"   -d '{
        "root":"data/images/test",
        "recursive":true,
        "tta":true,
        "return_predictions":true,
        "max_predictions":12
      }'
```

---

## Thresholds & Calibration
- Operating point is controlled by env var `TUMOR_THRESH` (default 0.22 if unset).  
- After evaluation, choose the threshold suggested by Youden-J (often **0.10** for RN34@224).  
- (Optional) Temperature scaling can be added to stabilize probabilities across datasets.

---

## Notes
- Designed for single-slice PNG/JPG and zipped DICOM series (center-slice heuristic).  
- Grad-CAM is available for CNN backbones (e.g., ResNet). Vision Transformers require a different explainer.

---

## License
MIT — see [LICENSE](LICENSE).

## Acknowledgements
- [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models), [FastAPI](https://fastapi.tiangolo.com/).

## Maintainer
**Deepesh Kumar Gupta**
