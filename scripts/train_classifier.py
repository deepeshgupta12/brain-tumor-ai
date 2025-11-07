import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import timm

# -----------------------
# Helpers
# -----------------------
SEED = 42
CLASS_NAMES = ["no_tumor", "glioma", "meningioma", "pituitary"]

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_pick():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

def make_transforms(img_size=224):
    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # [-1,1] range
    ])
    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    return train_tfms, val_tfms

class CSVDataset(Dataset):
    def __init__(self, csv_path, use_multiclass=True, transforms=None):
        df = pd.read_csv(csv_path)
        needed = {"path","binary_label","multiclass_label","class_name","case_id"}
        if not needed.issubset(df.columns):
            raise ValueError(f"{csv_path} missing columns: {needed - set(df.columns)}")
        self.paths = df["path"].tolist()
        self.y_multi = df["multiclass_label"].astype(int).tolist()
        self.y_bin = df["binary_label"].astype(int).tolist()
        self.use_multiclass = use_multiclass
        self.transforms = transforms

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = self.y_multi[idx] if self.use_multiclass else self.y_bin[idx]
        img = Image.open(p).convert("RGB")
        if self.transforms: img = self.transforms(img)
        return img, torch.tensor(y, dtype=torch.long)

def class_counts_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    cnts = df["multiclass_label"].value_counts().sort_index()
    # Ensure all 4 classes present in dict
    return {i:int(cnts.get(i,0)) for i in range(4)}

def compute_class_weights(counts_dict):
    # inverse frequency → normalize to mean=1.0
    counts = np.array([counts_dict[i] for i in range(4)], dtype=np.float32)
    counts[counts==0] = counts[counts>0].min()  # avoid div-by-zero
    inv = 1.0 / counts
    return torch.tensor(inv * (4.0 / inv.sum()), dtype=torch.float32)

def accuracy(pred, target):
    return (pred.argmax(1) == target).float().mean().item()

# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc  += accuracy(logits.detach().cpu(), y.detach().cpu()) * x.size(0)
        n += x.size(0)
    return total_loss/n, total_acc/n

@torch.inference_mode()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x,y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        total_acc  += accuracy(logits.detach().cpu(), y.detach().cpu()) * x.size(0)
        n += x.size(0)
    return total_loss/n, total_acc/n

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--out_dir", default="runs/cls_resnet18")
    ap.add_argument("--arch", default="resnet18", help="any timm model, e.g. resnet18/resnet50/convnext_tiny/vit_small_patch16_224")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    set_seed()
    device = device_pick()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_tfms, val_tfms = make_transforms(args.img_size)
    ds_tr = CSVDataset(args.train_csv, use_multiclass=True, transforms=train_tfms)
    ds_va = CSVDataset(args.val_csv,   use_multiclass=True, transforms=val_tfms)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, drop_last=False)

    # Model
    model = timm.create_model(args.arch, pretrained=True, num_classes=4, in_chans=3)
    model.to(device)

    # Loss with class weights (helps imbalance)
    counts = class_counts_from_csv(args.train_csv)
    weights = compute_class_weights(counts).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc, best_path = -1.0, None
    log_path = Path(args.out_dir) / "log.txt"
    with open(log_path, "w") as logf:
        logf.write(f"arch={args.arch} device={device} epochs={args.epochs}\n")
        logf.write(f"class_counts={counts} weights={weights.detach().cpu().tolist()}\n")

        for epoch in range(1, args.epochs+1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, dl_tr, loss_fn, optimizer, device)
            va_loss, va_acc = evaluate(model, dl_va, loss_fn, device)
            scheduler.step()
            dt = time.time()-t0
            msg = f"Epoch {epoch:03d} | {dt:5.1f}s | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}"
            print(msg); logf.write(msg+"\n"); logf.flush()

            # Save best
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_path = Path(args.out_dir) / "best_state.pt"
                torch.save({"state_dict": model.state_dict(),
                            "arch": args.arch,
                            "img_size": args.img_size,
                            "class_names": CLASS_NAMES}, best_path)

    # Also save metadata JSON for inference wiring
    meta = {
        "arch": args.arch,
        "img_size": args.img_size,
        "class_names": CLASS_NAMES,
        "train_counts": counts,
        "best_val_acc": float(best_val_acc)
    }
    with open(Path(args.out_dir) / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Convenience copy into ./models for the API (Step 7 will update loader)
    Path("models").mkdir(exist_ok=True)
    if best_path and best_path.exists():
        to_ckpt = Path("models/classifier_state.pt")
        to_meta = Path("models/metadata.json")
        import shutil
        shutil.copy2(best_path, to_ckpt)
        shutil.copy2(Path(args.out_dir) / "metadata.json", to_meta)
        print(f"Copied best checkpoint to {to_ckpt} and metadata to {to_meta}")

if __name__ == "__main__":
    main()
