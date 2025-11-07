import argparse, json, time, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import timm

# optional mixup/cutmix from timm
try:
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy
    HAS_TIMM_MIXUP = True
except Exception:
    HAS_TIMM_MIXUP = False

SEED = 42
CLASS_NAMES = ["no_tumor","glioma","meningioma","pituitary"]
N_CLASSES = 4

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_pick():
    if torch.backends.mps.is_available(): return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

def make_transforms(img_size=256):
    train_tfms = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.80, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        T.RandomErasing(p=0.1),
    ])
    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    return train_tfms, val_tfms

class CSVDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        df = pd.read_csv(csv_path)
        need = {"path","multiclass_label"}
        if not need.issubset(df.columns):
            raise ValueError(f"{csv_path} missing {need - set(df.columns)}")
        self.paths = df["path"].tolist()
        self.labels = df["multiclass_label"].astype(int).tolist()
        self.transforms = transforms
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        y = self.labels[i]
        img = Image.open(p).convert("RGB")
        if self.transforms: img = self.transforms(img)
        return img, torch.tensor(y, dtype=torch.long)

def class_counts_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    cnts = df["multiclass_label"].value_counts().sort_index()
    return {i:int(cnts.get(i,0)) for i in range(N_CLASSES)}

def compute_class_weights(counts_dict):
    c = np.array([counts_dict[i] for i in range(N_CLASSES)], dtype=np.float32)
    c[c==0] = c[c>0].min()
    inv = 1.0 / c
    return torch.tensor(inv * (N_CLASSES / inv.sum()), dtype=torch.float32)

def accuracy(logits, target):
    return (logits.argmax(1) == target).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--out_dir", default="runs/cls_rn50_256")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--use_mixup", type=lambda s: s.lower() in {"1","true","yes"}, default=False)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--cutmix", type=float, default=0.8)
    args = ap.parse_args()

    set_seed(); device = device_pick()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    tr_tf, va_tf = make_transforms(args.img_size)
    ds_tr = CSVDataset(args.train_csv, tr_tf)
    ds_va = CSVDataset(args.val_csv,   va_tf)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    model = timm.create_model(args.arch, pretrained=True, num_classes=N_CLASSES, in_chans=3).to(device)

    # loss / mixup setup
    counts = class_counts_from_csv(args.train_csv)
    class_w = compute_class_weights(counts).to(device)

    use_mix = bool(args.use_mixup and HAS_TIMM_MIXUP)
    if use_mix:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, prob=1.0,
            switch_prob=0.0, mode="batch", label_smoothing=0.0, num_classes=N_CLASSES
        )
        loss_train = SoftTargetCrossEntropy()   # with mixup, targets are soft
    else:
        mixup_fn = None
        loss_train = nn.CrossEntropyLoss(weight=class_w, label_smoothing=args.label_smoothing)

    loss_eval = nn.CrossEntropyLoss()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val_acc, best_path = -1.0, None
    with open(out/"log.txt", "w") as logf:
        logf.write(f"arch={args.arch} device={device} img={args.img_size} epochs={args.epochs}\n")
        logf.write(f"class_counts={counts} class_weights={class_w.detach().cpu().tolist()}\n")
        logf.write(f"mixup={'on' if use_mix else 'off'} m={args.mixup} c={args.cutmix} ls={args.label_smoothing}\n")

        for ep in range(1, args.epochs+1):
            t0 = time.time()
            # ---- train ----
            model.train()
            totL=totA=n=0
            for x,y in dl_tr:
                x=x.to(device); y=y.to(device)
                optim.zero_grad()
                if mixup_fn is not None:
                    x, y_soft = mixup_fn(x, y)
                    logits = model(x)
                    loss = loss_train(logits, y_soft)
                else:
                    logits = model(x)
                    loss = loss_train(logits, y)
                loss.backward()
                optim.step()
                bs = x.size(0)
                totL += loss.item()*bs
                if mixup_fn is None:
                    totA += accuracy(logits.detach().cpu(), y.detach().cpu())*bs
                n += bs
            tr_loss = totL/n
            tr_acc = (totA/n) if mixup_fn is None else float("nan")

            # ---- eval ----
            model.eval()
            totL=totA=n=0
            with torch.inference_mode():
                for x,y in dl_va:
                    x=x.to(device); y=y.to(device)
                    logits = model(x)
                    loss = loss_eval(logits, y)
                    bs = x.size(0)
                    totL += loss.item()*bs
                    totA += accuracy(logits.detach().cpu(), y.detach().cpu())*bs
                    n += bs
            va_loss = totL/n; va_acc = totA/n

            sched.step()
            msg = f"Epoch {ep:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}"
            print(msg); logf.write(msg+"\n"); logf.flush()

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_path = out/"best_state.pt"
                torch.save({"state_dict": model.state_dict(),
                            "arch": args.arch,
                            "img_size": args.img_size,
                            "class_names": CLASS_NAMES}, best_path)

    meta = {"arch": args.arch, "img_size": args.img_size, "class_names": CLASS_NAMES, "train_counts": counts, "best_val_acc": float(best_val_acc)}
    (out/"metadata.json").write_text(json.dumps(meta, indent=2))

    # convenience copy for API
    Path("models").mkdir(exist_ok=True)
    if best_path and best_path.exists():
        import shutil
        shutil.copy2(best_path, Path("models/classifier_state.pt"))
        shutil.copy2(out/"metadata.json", Path("models/metadata.json"))
        print(f"Copied best checkpoint to models/classifier_state.pt and metadata.json")
    print(f"Done. Best val acc = {best_val_acc:.4f}")
if __name__ == "__main__":
    main()
