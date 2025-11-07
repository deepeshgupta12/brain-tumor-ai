import argparse, os, re, csv, random, imghdr
from pathlib import Path
from typing import List, Tuple

ALIASES = {
  "no_tumor":"no_tumor","no-tumor":"no_tumor","notumor":"no_tumor",
  "glioma_tumor":"glioma","glioma":"glioma",
  "meningioma_tumor":"meningioma","meningioma_tomor":"meningioma","meningioma":"meningioma",
  "pituitary_tumor":"pituitary","pituitary":"pituitary",
}
CLASS_TO_IDX = {"no_tumor":0,"glioma":1,"meningioma":2,"pituitary":3}
VALID_EXT={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
CASE_REGEXES=[r"^(.*?)[_-]?slice\\d+$",r"^(.*?)[_-]?s\\d+$",r"^(.*?)[_-]?z\\d+$"]

def norm_class(folder:str)->str:
  key=folder.strip().lower().replace(" ","_"); return ALIASES.get(key,key)

def is_image(p:Path)->bool:
  return p.suffix.lower() in VALID_EXT and (imghdr.what(p) in {"png","jpeg","bmp","tiff"})

def case_id_from(stem:str)->str:
  for rx in CASE_REGEXES:
    m=re.match(rx,stem,re.IGNORECASE)
    if m: return m.group(1)
  return stem

def scan(root:Path)->List[Tuple[str,int,int,str,str]]:
  rows=[]
  for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
    cname=norm_class(sub.name)
    if cname not in CLASS_TO_IDX: continue
    mc=CLASS_TO_IDX[cname]; binlab=0 if cname=="no_tumor" else 1
    for f in sorted(sub.rglob("*")):
      if f.is_file() and is_image(f):
        rows.append((str(f),binlab,mc,cname,case_id_from(f.stem)))
  return rows

def strat_split_by_case(rows,val_frac=0.2,seed=42):
  random.seed(seed)
  class_cases={}
  for *_, mc, _, cid in rows:
    class_cases.setdefault(mc,set()).add(cid)
  tr_cases=set(); va_cases=set()
  for mc,cases in class_cases.items():
    cases=list(cases); random.shuffle(cases)
    k=max(1,int(round(len(cases)*(1-val_frac))))
    tr_cases.update(cases[:k]); va_cases.update(cases[k:])
  tr=[]; va=[]
  for r in rows:
    (tr if r[-1] in tr_cases and r[-1] not in va_cases else va).append(r)
  return tr,va

def write_csv(path:Path, rows):
  path.parent.mkdir(parents=True,exist_ok=True)
  with open(path,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["path","binary_label","multiclass_label","class_name","case_id"]); w.writerows(rows)

def main():
  ap=argparse.ArgumentParser()
  ap.add_argument("--root",required=True); ap.add_argument("--out",default="data/folder_csv")
  ap.add_argument("--val_frac",type=float,default=0.2); ap.add_argument("--seed",type=int,default=42)
  a=ap.parse_args()
  rows=scan(Path(a.root))
  if not rows: raise SystemExit("No images found under --root.")
  tr,va=strat_split_by_case(rows,a.val_frac,a.seed)
  out=Path(a.out); write_csv(out/"train.csv",tr); write_csv(out/"val.csv",va)
  from collections import Counter
  print(f"Total {len(rows)} | Train {len(tr)} | Val {len(va)}")
  print("Train dist:",Counter([r[2] for r in tr])); print("Val dist:",Counter([r[2] for r in va]))
  print(f"CSV written to {out}")
if __name__=="__main__": main()
