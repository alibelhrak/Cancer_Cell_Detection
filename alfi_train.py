import os, sys, json, time, argparse, warnings, math
import numpy as np
from itertools import cycle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
from sklearn.model_selection import GroupShuffleSplit
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from torch.amp import GradScaler, autocast
    AMP_DEVICE_ARG = {"device_type": "cuda"}
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AMP_DEVICE_ARG = {}

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import pandas as pd
from PIL import Image
from sklearn.metrics import (  
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    DATA_ROOT = (
        "/mnt/projects/sutravek_project/Ali_belhrak/"
        "SI_Project/ALFIdatasetFinal/Data&Annotations"
    )
    SAVE_DIR    = "./newcheckpoints2"
    RESULTS_DIR = "./newresults2"

    SEQ_LEN     = 8
    IMG_SIZE    = 224
    BATCH_SIZE  = 8
    NUM_WORKERS = 4
    CROP_CELLS  = True
    N_ACCUM     = 4

    T1_SPLIT_TRAIN = 0.70
    T1_SPLIT_VAL   = 0.15
    T1_SPLIT_TEST  = 0.15

    SPLIT_SEED = 42

    # Model
    LSTM_HIDDEN  = 256
    LSTM_LAYERS  = 1
    LSTM_DROPOUT = 0.5
    CNN_DROPOUT  = 0.5
    WEIGHT_DECAY = 1e-2

    # Training
    EPOCHS            = 35
    LR                = 1e-4
    WARMUP_EPOCHS     = 3
    FREEZE_CNN_EPOCHS = 3
    PHASE_A_EPOCHS    = 8

    TASK1_LOSS_WEIGHT_A = 1.0
    TASK1_LOSS_WEIGHT_B = 1.0

    PATIENCE  = 8
    USE_AMP   = True

    MIN_TRACK_PURITY  = 0.6
    MIN_WINDOW_PURITY = 0.6


TASK1_CLASSES = {"Mitosis": 0, "Interphase": 1}
TASK1_NAMES   = {v: k for k, v in TASK1_CLASSES.items()}

SEQ_FAMILIES = {
    "MI": ["MI01","MI02","MI03","MI04","MI05","MI06","MI07","MI08"]
}


# ─────────────────────────────────────────────────────────────────────────────
#  SEQUENCE-LEVEL STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def stratified_sequence_split(
    seq_names:   List[str],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[set, set, set]:
    rng = np.random.default_rng(seed)
    family_map: Dict[str, List[str]] = defaultdict(list)
    for name in seq_names:
        assigned = False
        for prefix, members in SEQ_FAMILIES.items():
            if name in members:
                family_map[prefix].append(name)
                assigned = True
                break
        if not assigned:
            family_map["OTHER"].append(name)

    train_set, val_set, test_set = set(), set(), set()
    for family, members in family_map.items():
        arr  = np.array(sorted(members))
        rng.shuffle(arr)
        n    = len(arr)
        n_tr = max(1, int(n * train_ratio))
        n_vl = max(1, int(n * val_ratio))
        if n - n_tr - n_vl < 1 and n >= 3:
            n_vl = max(1, n - n_tr - 1)
        train_set.update(arr[:n_tr])
        val_set.update(arr[n_tr: n_tr + n_vl])
        test_set.update(arr[n_tr + n_vl:])

    assert not (train_set & val_set)
    assert not (train_set & test_set)
    assert not (val_set   & test_set)

    print(f"\n  Sequence split over {len(seq_names)} seqs:")
    for family in sorted(family_map):
        tr = sorted(s for s in family_map[family] if s in train_set)
        vl = sorted(s for s in family_map[family] if s in val_set)
        te = sorted(s for s in family_map[family] if s in test_set)
        print(f"    {family:6s}  train={tr}  val={vl}  test={te}")
    return train_set, val_set, test_set


# ─────────────────────────────────────────────────────────────────────────────
#  CELL-FAMILY SPLIT (Task 1 only)
# ─────────────────────────────────────────────────────────────────────────────

def cell_family_split(
    df:          pd.DataFrame,
    label_map:   Dict[str, int],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[Set[float], Set[float], Set[float]]:
    """Group IDs by floor(ID) so parent + daughters go to the same split."""
    rng = np.random.default_rng(seed)

    family_root: Dict[int, Set[float]] = defaultdict(set)
    for raw_id in df["ID"].unique():
        root = int(math.floor(raw_id))
        family_root[root].add(float(raw_id))

    def _family_class(root: int) -> int:
        rows = df[df["ID"] == float(root)]
        if rows.empty:
            rows = df[df["ID"].apply(lambda x: int(math.floor(x))) == root]
        if rows.empty:
            return -1
        return label_map.get(rows["Class"].value_counts().index[0], -1)

    class_bins: Dict[int, List[int]] = defaultdict(list)
    for root in sorted(family_root):
        class_bins[_family_class(root)].append(root)

    train_ids: Set[float] = set()
    val_ids:   Set[float] = set()
    test_ids:  Set[float] = set()

    for cls, roots in class_bins.items():
        arr  = np.array(sorted(roots))
        rng.shuffle(arr)
        n    = len(arr)
        n_tr = max(1, int(n * train_ratio))
        n_vl = max(1, int(n * val_ratio))
        if n - n_tr - n_vl < 1 and n >= 3:
            n_vl = max(1, n - n_tr - 1)
        for r in arr[:n_tr]:
            train_ids.update(family_root[int(r)])
        for r in arr[n_tr: n_tr + n_vl]:
            val_ids.update(family_root[int(r)])
        for r in arr[n_tr + n_vl:]:
            test_ids.update(family_root[int(r)])

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids   & test_ids)
    return train_ids, val_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.93,1.07)),
            transforms.ToTensor(), norm,
        ])
    return transforms.Compose([transforms.Resize((img_size, img_size)),
                                transforms.ToTensor(), norm])


# ─────────────────────────────────────────────────────────────────────────────
#  ANNOTATION LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_annotations(seq_dir: str) -> pd.DataFrame:
    name = os.path.basename(seq_dir)
    p = os.path.join(seq_dir, f"{name}_DTLTruth.csv")
    if os.path.exists(p):
        df = pd.read_csv(p); df.columns = df.columns.str.strip()
        return df[df["Class"].isin(TASK1_CLASSES)].reset_index(drop=True)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  TRACK BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_cell_tracks(
    seq_dir:           str,
    seq_len:           int,
    split:             str,
    train_ratio:       float = 0.70,
    val_ratio:         float = 0.15,
    seed:              int   = 42,
    min_track_purity:  float = 0.75,
    min_window_purity: float = 0.75,
    use_family_split:  bool  = True,
) -> List[Dict]:
    seq_name  = os.path.basename(seq_dir)
    image_dir = os.path.join(seq_dir, "Images")
    df        = load_annotations(seq_dir)
    if df.empty or not os.path.isdir(image_dir):
        return []

    label_map = TASK1_CLASSES

    if use_family_split and split != "all":
        tr_ids, vl_ids, te_ids = cell_family_split(
            df, label_map, train_ratio, val_ratio, seed
        )
        active_ids: Optional[Set[float]] = {
            "train": tr_ids, "val": vl_ids, "test": te_ids
        }[split]
    else:
        active_ids = None
    samples = []

    for cell_id, cdf in df.groupby("ID"):
        if active_ids is not None and float(cell_id) not in active_ids:
            continue

        cdf = cdf.sort_values("ImNo").reset_index(drop=True)
        cdf = cdf[cdf["Class"].isin(label_map)]
        if cdf.empty:
            continue

        lc           = cdf["Class"].value_counts()
        top_label    = lc.index[0]
        top_fraction = lc.iloc[0] / len(cdf)
        label_int    = label_map[top_label]

        if top_fraction < min_track_purity:
            continue

        frame_nums  = cdf["ImNo"].tolist()
        frame_paths = [
            os.path.join(image_dir, f"I_{seq_name}_{fn:04d}.png")
            for fn in frame_nums
        ]
        bboxes       = list(zip(cdf["xmin"], cdf["ymin"],
                                cdf["width"], cdf["height"]))
        class_labels = cdf["Class"].tolist()

        valid = [(p,b,c) for p,b,c in zip(frame_paths,bboxes,class_labels)
                 if os.path.exists(p)]
        if len(valid) < 2:
            continue
        frame_paths, bboxes, class_labels = zip(*valid)
        frame_paths  = list(frame_paths)
        bboxes       = list(bboxes)
        class_labels = list(class_labels)

        def _ok(cls_list, lbl):
            wl  = [label_map.get(c, -1) for c in cls_list]
            dom = max(set(wl), key=wl.count)
            return wl.count(dom) / len(wl) >= min_window_purity

        if len(frame_paths) < seq_len:
            pad = seq_len - len(frame_paths)
            fp  = frame_paths  + [frame_paths[-1]]  * pad
            bb  = bboxes       + [bboxes[-1]]       * pad
            cl  = class_labels + [class_labels[-1]] * pad
            if _ok(cl, label_int):
                samples.append(dict(frames=fp, bboxes=bb, label=label_int,
                                    cell_id=cell_id, seq_name=seq_name))
        else:
            stride = max(1, (len(frame_paths) - seq_len) // 3)
            for s in range(0, len(frame_paths) - seq_len + 1, stride):
                if _ok(class_labels[s:s+seq_len], label_int):
                    samples.append(dict(
                        frames=frame_paths[s:s+seq_len],
                        bboxes=bboxes[s:s+seq_len],
                        label=label_int, cell_id=cell_id,
                        seq_name=seq_name,
                    ))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ALFIDatasetFromSamples(Dataset):
    def __init__(self, samples: List[Dict], split: str,
                 img_size: int = 224, crop_cells: bool = True):
        self.samples    = samples
        self.split      = split
        self.crop_cells = crop_cells
        self.transform  = get_transforms(split, img_size)

        dist     = Counter(s["label"] for s in self.samples)
        dist_str = "  ".join(f"{TASK1_NAMES[k]}:{v}" for k,v in sorted(dist.items()))
        print(f"  [Task 1 | {split:5s}]  {len(self.samples):5d} samples  |  {dist_str}")

    def _load_frame(self, path: str, bbox=None) -> Image.Image:
        # identical to ALFIDataset._load_frame
        try:
            img = Image.open(path)
        except Exception:
            return Image.new("RGB", (224, 224))
        arr = np.array(img, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        if bbox is not None and self.crop_cells:
            xmin, ymin, w, h = [int(v) for v in bbox]
            pad = int(max(w, h) * 0.20)
            x0 = max(0, xmin-pad); y0 = max(0, ymin-pad)
            x1 = min(img.width, xmin+w+pad); y1 = min(img.height, ymin+h+pad)
            if x1 > x0 and y1 > y0:
                img = img.crop((x0, y0, x1, y1))
        return img

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = torch.stack([self.transform(self._load_frame(p, b))
                               for p, b in zip(s["frames"], s["bboxes"])])
        return frames, s["label"]

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_weighted_loader(ds, cfg, drop_last=True):
    lc = Counter(s["label"] for s in ds.samples)
    sw = [1.0 / lc[s["label"]**0.5] for s in ds.samples]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                      num_workers=cfg.NUM_WORKERS,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=drop_last,
                      persistent_workers=(cfg.NUM_WORKERS > 0))

def _make_loader(ds, cfg, shuffle=False):
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
                      num_workers=cfg.NUM_WORKERS,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=(cfg.NUM_WORKERS > 0))


def build_dataloaders(cfg: Config) -> Dict:
    print("\n" + "═"*65)
    print(f"  Loading data  {cfg.DATA_ROOT}")
    print("═"*65)

    loaders: Dict = {}

    t1_seqs = [e for e in sorted(os.listdir(cfg.DATA_ROOT))
               if os.path.isdir(os.path.join(cfg.DATA_ROOT, e, "Images"))
               and e.upper().startswith("MI")]
    if not t1_seqs:
        print("[WARNING] No MI sequences found for Task 1 — check DATA_ROOT")

    print(f"\n── Task 1: {len(t1_seqs)} sequences ──")

    # Build all samples first across all sequences
    all_samples = []
    for seq_name in t1_seqs:
        seq_dir = os.path.join(cfg.DATA_ROOT, seq_name)
        all_samples.extend(build_cell_tracks(
            seq_dir=seq_dir, seq_len=cfg.SEQ_LEN, split="all",
            train_ratio=cfg.T1_SPLIT_TRAIN, val_ratio=cfg.T1_SPLIT_VAL,
            seed=cfg.SPLIT_SEED,
            min_track_purity=cfg.MIN_TRACK_PURITY,
            min_window_purity=cfg.MIN_WINDOW_PURITY,
            use_family_split=False,
        ))

    if not all_samples:
        print("[ERROR] No samples found"); return loaders

    labels  = np.array([s["label"]   for s in all_samples])
    groups  = np.array([f"{s['seq_name']}_{s['cell_id']}" for s in all_samples])
    indices = np.arange(len(all_samples))

    # Test split
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.T1_SPLIT_TEST,
                            random_state=cfg.SPLIT_SEED)
    train_val_idx, test_idx = next(gss.split(indices, labels, groups))

    # Val split from remaining
    gss2 = GroupShuffleSplit(n_splits=1,
                             test_size=cfg.T1_SPLIT_VAL / (1 - cfg.T1_SPLIT_TEST),
                             random_state=cfg.SPLIT_SEED)
    labels_tv = labels[train_val_idx]
    groups_tv = groups[train_val_idx]
    tr_rel, val_rel = next(gss2.split(train_val_idx, labels_tv, groups_tv))
    train_idx = train_val_idx[tr_rel]
    val_idx   = train_val_idx[val_rel]

    print(f"  Split → train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        n0 = int((labels[idx] == 0).sum())
        n1 = int((labels[idx] == 1).sum())
        print(f"    {name:5s}  Mitosis={n0}  Interphase={n1}")

    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}

    loaders["task1"] = {}
    for split in ["train", "val", "test"]:
        idx = split_indices[split]
        samples = [all_samples[i] for i in idx]
        ds = ALFIDatasetFromSamples(samples, split, cfg.IMG_SIZE, cfg.CROP_CELLS)
        loaders["task1"][split] = (
            _make_weighted_loader(ds, cfg) if split == "train"
            else _make_loader(ds, cfg)
        )

    # Class weights
    tr_labels = labels[train_idx]
    classes = sorted(set(tr_labels.tolist()))
    w = compute_class_weight("balanced", classes=np.array(classes),
                             y=tr_labels)
    loaders["task1_weights"] = torch.tensor([5.0, 1.0], dtype=torch.float)
    print(f"\n  Task1 weights: "
          f"{ {TASK1_NAMES[c]: f'{w[i]:.3f}' for i,c in enumerate(classes)} }")

    return loaders

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetB0_BiLSTM(nn.Module):
    def __init__(self, num_classes_t1=2,
                 lstm_hidden=256, lstm_layers=1, lstm_drop=0.5, cnn_drop=0.5):
        super().__init__()
        bb = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cnn_features = bb.features
        self.cnn_avgpool  = bb.avgpool
        self.cnn_drop     = nn.Dropout(cnn_drop)
        CNN_DIM = 1280

        self.lstm = nn.LSTM(CNN_DIM, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=lstm_drop if lstm_layers>1 else 0.0)
        LSTM_DIM = lstm_hidden * 2

        self.attention = nn.Sequential(
            nn.Linear(LSTM_DIM, 128), nn.Tanh(), nn.Linear(128, 1))

        self.head_t1 = nn.Sequential(
            nn.LayerNorm(LSTM_DIM), nn.Dropout(0.40),
            nn.Linear(LSTM_DIM, 64), nn.ReLU(True),
            nn.Dropout(0.20), nn.Linear(64, num_classes_t1))

        self._init_weights()

    def _init_weights(self):
        for n, p in self.lstm.named_parameters():
            if   "weight_ih" in n: nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in n: nn.init.orthogonal_(p.data)
            elif "bias"      in n:
                p.data.fill_(0); k = p.size(0)
                p.data[k//4:k//2].fill_(1.0)
        for mod in [self.head_t1, self.attention]:
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def freeze_cnn(self):
        for p in self.cnn_features.parameters(): p.requires_grad = False
    def unfreeze_cnn(self):
        for p in self.cnn_features.parameters(): p.requires_grad = True

    def _cnn(self, x):
        return self.cnn_drop(self.cnn_avgpool(self.cnn_features(x))).flatten(1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats       = self._cnn(x.view(B*T,C,H,W)).view(B,T,-1)
        lstm_out, _ = self.lstm(feats)
        attn_w      = torch.softmax(self.attention(lstm_out), dim=1)
        context     = (attn_w * lstm_out).sum(dim=1)
        return self.head_t1(context), attn_w.squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
#  FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__(); self.gamma = gamma; self.weight = weight
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets,
                                         weight=self.weight, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(opt, warmup, total, min_f=0.01):
    def _lam(ep):
        if ep < warmup: return (ep+1) / max(warmup,1)
        prog = (ep-warmup) / max(total-warmup+1,1)
        return max(0.5*(1+np.cos(np.pi*prog)), min_f)
    return optim.lr_scheduler.LambdaLR(opt, _lam)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model, loaders, cfg, device):
        self.model  = model.to(device)
        self.loaders= loaders
        self.cfg    = cfg
        self.device = device
        self.use_amp= cfg.USE_AMP and device.type == "cuda"

        os.makedirs(cfg.SAVE_DIR,    exist_ok=True)
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

        t1w = loaders.get("task1_weights")
        self.crit_t1 = FocalLoss(1.0, t1w.to(device) if t1w is not None else None)

        cnn_p   = list(self.model.cnn_features.parameters())
        other_p = [p for p in model.parameters()
                   if not any(p is c for c in cnn_p)]
        self.optimizer = optim.AdamW(
            [{"params": cnn_p,   "lr": cfg.LR*0.1},
             {"params": other_p, "lr": cfg.LR}],
            weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = build_scheduler(self.optimizer, cfg.WARMUP_EPOCHS, cfg.EPOCHS)

        self.plateau = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-7)

        self.scaler = GradScaler() if self.use_amp else None

        self.history = {k:[] for k in
            ["train_loss","val_loss","val_acc_t1","train_acc_t1",
             "val_f1_t1","val_f1_t1_macro"]}
        self.best_f1   = -1.0
        self.pat_count = 0
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 5))
        self.fig.suptitle("Live Training", fontsize=14)
        plt.show()
        
    def _update_plot(self):
        ep = range(1, len(self.history["train_loss"]) + 1)
        titles = ["Focal Loss", "Val Accuracy", "Val F1"]
        data = [
            [(self.history["train_loss"], "Train", _D["a1"]),
             (self.history["val_loss"],   "Val",   _D["a2"])],
            [(self.history["val_acc_t1"],   "Val",   _D["a1"]),
             (self.history["train_acc_t1"], "Train", _D["a2"])],
            [(self.history["val_f1_t1_macro"], "macro",    _D["a1"]),
             (self.history["val_f1_t1"],       "weighted", _D["a2"])],
        ]
        self.fig.patch.set_facecolor(_D["bg"])
        for ax, title, series in zip(self.axes, titles, data):
            ax.clear()
            ax.set_facecolor(_D["panel"])
            ax.tick_params(colors=_D["text"])
            ax.xaxis.label.set_color(_D["text"])
            ax.yaxis.label.set_color(_D["text"])
            ax.title.set_color(_D["text"])
            for spine in ax.spines.values():
                spine.set_edgecolor(_D["grid"])
            ax.grid(True, color=_D["grid"], linestyle="--", alpha=0.5)
            for vals, label, color in series:
                ax.plot(ep, vals, label=label, color=color, lw=2)
            ax.set_title(title)
            legend = ax.legend()
            for text in legend.get_texts():
                text.set_color(_D["text"])
            legend.get_frame().set_facecolor(_D["panel"])
            legend.get_frame().set_edgecolor(_D["grid"])
            ax.set_xlabel("Epoch")
        self.fig.tight_layout()        # ← 8 spaces
        self.fig.canvas.draw()         # ← 8 spaces
        self.fig.canvas.flush_events() # ← 8 spaces
        out = os.path.join(self.cfg.RESULTS_DIR, "training_curves_live.png")
        self.fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=_D["bg"])

    def _get_w(self, ep):
        if ep <= self.cfg.PHASE_A_EPOCHS:
            return self.cfg.TASK1_LOSS_WEIGHT_A
        return self.cfg.TASK1_LOSS_WEIGHT_B

    def _forward(self, batch, w1):
        fr, lb = batch
        fr = fr.to(self.device, non_blocking=True)
        lb = lb.to(self.device, non_blocking=True)
        logits, _ = self.model(fr)
        loss = w1 * self.crit_t1(logits, lb)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().tolist()
        labels = lb.cpu().tolist()
        return loss, (preds, labels)

    def _epoch(self, split, epoch=0):
        is_tr = split == "train"
        self.model.train() if is_tr else self.model.eval()

        loader = self.loaders.get("task1", {}).get(split)
        if loader is None:
            return {"loss":0, "acc_t1":0, "f1_t1":0, "f1_t1_macro":0,
                    "preds_t1":([],[])}

        w1 = self._get_w(epoch)
        total = 0.0
        all_preds, all_labels = [], []

        accum    = self.cfg.N_ACCUM if is_tr else 1
        step_num = 0

        if is_tr:
            self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            with torch.set_grad_enabled(is_tr):
                if is_tr and self.use_amp:
                    with autocast(**AMP_DEVICE_ARG):
                        loss, (preds, labels) = self._forward(batch, w1)
                else:
                    loss, (preds, labels) = self._forward(batch, w1)

                if is_tr:
                    loss_scaled = loss / accum
                    if self.scaler:
                        self.scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()

                    step_num += 1
                    if step_num % accum == 0 or step == len(loader)-1:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

            total += loss.item()
            all_preds.extend(preds)
            all_labels.extend(labels)

        n   = len(loader)
        avg = total / max(n, 1)
        acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0) if all_labels else 0.0
        f1_macro    = f1_score(all_labels, all_preds, average="macro",    zero_division=0) if all_labels else 0.0

        return {
            "loss":         avg,
            "acc_t1":       acc,
            "f1_t1":        f1_weighted,
            "f1_t1_macro":  f1_macro,
            "preds_t1":     (all_preds, all_labels),
        }

    def train(self):
        cfg = self.cfg
        print(f"\n{'═'*65}")
        print(f"  EfficientNetB0+BiLSTM  ALFI — Task 1 Only")
        print(f"  Device: {self.device}  AMP: {self.use_amp}")
        print(f"  Gradient accumulation: {cfg.N_ACCUM} steps (eff. batch={cfg.BATCH_SIZE*cfg.N_ACCUM})")
        print(f"  Early stopping: macro F1 (Task 1)")
        print(f"  Phase A ep1-{cfg.PHASE_A_EPOCHS}: T1×{cfg.TASK1_LOSS_WEIGHT_A}")
        print(f"  Phase B ep{cfg.PHASE_A_EPOCHS+1}+: T1×{cfg.TASK1_LOSS_WEIGHT_B}")
        print(f"{'═'*65}\n")

        self.model.freeze_cnn()
        print(f"[Phase 1] CNN frozen for {cfg.FREEZE_CNN_EPOCHS} epochs")

        for ep in range(1, cfg.EPOCHS+1):
            t0 = time.time()
            if ep == cfg.FREEZE_CNN_EPOCHS+1:
                self.model.unfreeze_cnn()
                print(f"\n[Phase 2] CNN unfrozen\n")

            lr  = self.optimizer.param_groups[1]["lr"]
            tr  = self._epoch("train", ep)
            val = self._epoch("val",   ep)
            self.scheduler.step()

            cf1 = val["f1_t1_macro"]

            if ep > cfg.WARMUP_EPOCHS:
                self.plateau.step(cf1)

            for k, v in [
                ("train_loss",    tr["loss"]),
                ("val_loss",      val["loss"]),
                ("val_acc_t1",    val["acc_t1"]),
                ("train_acc_t1",  tr["acc_t1"]),
                ("val_f1_t1",     val["f1_t1"]),
                ("val_f1_t1_macro", val["f1_t1_macro"]),
            ]:
                self.history[k].append(v)
            self._update_plot()
            print(f"Ep {ep:3d}/{cfg.EPOCHS}  lr={lr:.2e}  "
                  f"tr={tr['loss']:.4f}  val={val['loss']:.4f}  "
                  f"acc={val['acc_t1']:.3f}  "
                  f"f1={val['f1_t1_macro']:.3f}(macro)  "
                  f"f1={val['f1_t1']:.3f}(weighted)  "
                  f"({time.time()-t0:.1f}s)")

            if cf1 > self.best_f1:
                self.best_f1 = cf1; self.pat_count = 0
                self._save("best", ep, val)
                print(f"  ✓ Best (macro_f1={cf1:.4f})")
            else:
                self.pat_count += 1
                if self.pat_count >= cfg.PATIENCE:
                    print(f"\n  Early stop at ep {ep}"); break

        with open(os.path.join(cfg.SAVE_DIR, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        plt.ioff()       
        plt.close(self.fig) 
    def evaluate(self, split="test"):
        print(f"\n{'═'*65}\n  Evaluating [{split}]\n{'═'*65}")
        ckpt = os.path.join(self.cfg.SAVE_DIR, "checkpoint_best.pt")
        if os.path.exists(ckpt):
            d = torch.load(ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(d["model_state_dict"])
            print(f"  Loaded epoch {d['epoch']}\n")
        else:
            print("  [WARNING] No checkpoint\n")

        m = self._epoch(split, self.cfg.EPOCHS)
        p1, l1 = m["preds_t1"]

        if l1:
            print("── Task 1 ──────────────────────────────────────────────")
            print(classification_report(l1, p1,
                  labels=list(range(len(TASK1_CLASSES))),
                  target_names=[TASK1_NAMES[i] for i in sorted(TASK1_NAMES)],
                  zero_division=0))
            print(confusion_matrix(l1, p1, labels=list(range(len(TASK1_CLASSES)))), "\n")
        return m

    def _save(self, tag, epoch, metrics):
        torch.save({"epoch": epoch,
                    "model_state_dict":     self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    **{k: metrics[k] for k in
                       ["loss", "acc_t1", "f1_t1", "f1_t1_macro"]}},
                   os.path.join(self.cfg.SAVE_DIR, f"checkpoint_{tag}.pt"))

    def load_checkpoint(self, path):
        d = torch.load(path, map_location=self.device)
        self.model.load_state_dict(d["model_state_dict"])
        self.optimizer.load_state_dict(d["optimizer_state_dict"])
        print(f"Loaded {path} (epoch {d['epoch']})")
        return d["epoch"]


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

_D = {"bg":"#0f1117","panel":"#1a1e2e","a1":"#4fc3f7","a2":"#f48fb1",
      "a3":"#a5d6a7","a4":"#ffcc02","text":"#e0e0e0","grid":"#2a2a3e"}

def _style():
    plt.rcParams.update({
        "figure.facecolor": _D["bg"],
        "axes.facecolor":   _D["panel"],
        "axes.edgecolor":   _D["grid"],
        "axes.labelcolor":  _D["text"],
        "xtick.color":      _D["text"],
        "ytick.color":      _D["text"],
        "text.color":       _D["text"],
        "grid.color":       _D["grid"],
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
    })

def plot_curves(history, save_dir):
    _style()
    ep = range(1, len(history["train_loss"])+1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("EfficientNetB0+BiLSTM — Task 1 Only",
                 fontsize=14, color=_D["text"], fontweight="bold")
    c = [_D["a1"], _D["a2"], _D["a3"], _D["a4"]]

    axes[0].plot(ep, history["train_loss"], color=c[0], lw=2, label="Train")
    axes[0].plot(ep, history["val_loss"],   color=c[1], lw=2, ls="--", label="Val")
    axes[0].set_title("Focal Loss"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(ep, history["val_acc_t1"], color=c[0], lw=2, label="Task 1")
    axes[1].set_title("Val Accuracy"); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(ep, history["val_f1_t1_macro"], color=c[0], lw=2,       label="T1 macro")
    axes[2].plot(ep, history["val_f1_t1"],       color=c[1], lw=1.5, ls=":", label="T1 weighted")
    axes[2].set_title("Val F1 (macro solid, weighted dotted)")
    axes[2].set_ylim(0, 1); axes[2].legend(fontsize=8); axes[2].grid(True)

    for ax in axes: ax.set_xlabel("Epoch")
    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_D["bg"])
    plt.close(); print(f"  Curves → {out}")

def plot_cm(yt, yp, names, title, path, labels=None):
    _style()
    if labels is None: labels = list(range(len(names)))
    cm  = confusion_matrix(yt, yp, labels=labels)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    n   = len(names)
    fig, ax = plt.subplots(figsize=(max(6, n*1.8), max(5, n*1.5)))
    sns.heatmap(cmn, annot=cm, fmt="d",
                cmap=sns.color_palette("Blues", as_cmap=True),
                xticklabels=names, yticklabels=names,
                linewidths=0.5, linecolor=_D["grid"],
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=_D["bg"])
    plt.close(); print(f"  CM → {path}")

def generate_plots(cfg, tm):
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    h = os.path.join(cfg.SAVE_DIR, "history.json")
    if os.path.exists(h):
        with open(h) as f: plot_curves(json.load(f), cfg.RESULTS_DIR)
    p1, l1 = tm.get("preds_t1", ([], []))
    if l1:
        plot_cm(l1, p1, ["Mitosis", "Interphase"],
                "Task 1 — Interphase vs Mitosis",
                os.path.join(cfg.RESULTS_DIR, "cm_task1.png"), labels=[0,1])
    save_metrics_csv(cfg, tm, split="test")   

def save_metrics_csv(cfg, tm, split="test"):
    p1, l1 = tm.get("preds_t1", ([], []))
    if not l1:
        print("  [SKIP] No predictions to save.")
        return

    from sklearn.metrics import precision_score, recall_score

    # Per-class metrics
    precision = precision_score(l1, p1, average=None, labels=[0,1], zero_division=0)
    recall    = recall_score(l1, p1, average=None, labels=[0,1], zero_division=0)
    f1_per    = f1_score(l1, p1, average=None, labels=[0,1], zero_division=0)

    # Overall metrics
    acc          = accuracy_score(l1, p1)
    f1_macro     = f1_score(l1, p1, average="macro",    zero_division=0)
    f1_weighted  = f1_score(l1, p1, average="weighted", zero_division=0)
    f1_mitosis   = f1_per[1]

    # Confusion matrix values
    cm = confusion_matrix(l1, p1, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    rows = [
        # Per-class rows
        {"Metric": "Precision",      "Class": "Interphase", "Value": round(precision[0], 4)},
        {"Metric": "Precision",      "Class": "Mitosis",    "Value": round(precision[1], 4)},
        {"Metric": "Recall",         "Class": "Interphase", "Value": round(recall[0], 4)},
        {"Metric": "Recall",         "Class": "Mitosis",    "Value": round(recall[1], 4)},
        {"Metric": "F1 Score",       "Class": "Interphase", "Value": round(f1_per[0], 4)},
        {"Metric": "F1 Score",       "Class": "Mitosis",    "Value": round(f1_per[1], 4)},
        # Overall rows
        {"Metric": "Accuracy",       "Class": "Overall",    "Value": round(acc, 4)},
        {"Metric": "F1 Macro",       "Class": "Overall",    "Value": round(f1_macro, 4)},
        {"Metric": "F1 Weighted",    "Class": "Overall",    "Value": round(f1_weighted, 4)},
        # Confusion matrix
        {"Metric": "True Positive",  "Class": "Mitosis",    "Value": int(tp)},
        {"Metric": "True Negative",  "Class": "Interphase", "Value": int(tn)},
        {"Metric": "False Positive", "Class": "Interphase→Mitosis", "Value": int(fp)},
        {"Metric": "False Negative", "Class": "Mitosis→Interphase", "Value": int(fn)},
        # Counts
        {"Metric": "Test Samples",   "Class": "Interphase", "Value": int(tn + fp)},
        {"Metric": "Test Samples",   "Class": "Mitosis",    "Value": int(tp + fn)},
        {"Metric": "Test Samples",   "Class": "Total",      "Value": int(len(l1))},
    ]

    df = pd.DataFrame(rows)
    out = os.path.join(cfg.RESULTS_DIR, f"metrics_{split}.csv")
    df.to_csv(out, index=False)
    print(f"  Metrics → {out}")
# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cfg = Config()
    p.add_argument("--data_root",           default=cfg.DATA_ROOT)
    p.add_argument("--save_dir",            default=cfg.SAVE_DIR)
    p.add_argument("--results_dir",         default=cfg.RESULTS_DIR)
    p.add_argument("--seq_len",             type=int,   default=cfg.SEQ_LEN)
    p.add_argument("--img_size",            type=int,   default=cfg.IMG_SIZE)
    p.add_argument("--batch_size",          type=int,   default=cfg.BATCH_SIZE)
    p.add_argument("--num_workers",         type=int,   default=cfg.NUM_WORKERS)
    p.add_argument("--n_accum",             type=int,   default=cfg.N_ACCUM)
    p.add_argument("--no_crop",             action="store_true")
    p.add_argument("--lstm_hidden",         type=int,   default=cfg.LSTM_HIDDEN)
    p.add_argument("--lstm_layers",         type=int,   default=cfg.LSTM_LAYERS)
    p.add_argument("--lstm_dropout",        type=float, default=cfg.LSTM_DROPOUT)
    p.add_argument("--epochs",              type=int,   default=cfg.EPOCHS)
    p.add_argument("--lr",                  type=float, default=cfg.LR)
    p.add_argument("--weight_decay",        type=float, default=cfg.WEIGHT_DECAY)
    p.add_argument("--warmup_epochs",       type=int,   default=cfg.WARMUP_EPOCHS)
    p.add_argument("--freeze_cnn_epochs",   type=int,   default=cfg.FREEZE_CNN_EPOCHS)
    p.add_argument("--patience",            type=int,   default=cfg.PATIENCE)
    p.add_argument("--phase_a_epochs",      type=int,   default=cfg.PHASE_A_EPOCHS)
    p.add_argument("--no_amp",              action="store_true")
    p.add_argument("--min_track_purity",    type=float, default=cfg.MIN_TRACK_PURITY)
    p.add_argument("--min_window_purity",   type=float, default=cfg.MIN_WINDOW_PURITY)
    p.add_argument("--test_only",           action="store_true")
    p.add_argument("--checkpoint",          type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.DATA_ROOT         = args.data_root
    cfg.SAVE_DIR          = args.save_dir
    cfg.RESULTS_DIR       = args.results_dir
    cfg.SEQ_LEN           = args.seq_len
    cfg.IMG_SIZE          = args.img_size
    cfg.BATCH_SIZE        = args.batch_size
    cfg.NUM_WORKERS       = args.num_workers
    cfg.N_ACCUM           = args.n_accum
    cfg.CROP_CELLS        = not args.no_crop
    cfg.LSTM_HIDDEN       = args.lstm_hidden
    cfg.LSTM_LAYERS       = args.lstm_layers
    cfg.LSTM_DROPOUT      = args.lstm_dropout
    cfg.EPOCHS            = args.epochs
    cfg.LR                = args.lr
    cfg.WEIGHT_DECAY      = args.weight_decay
    cfg.WARMUP_EPOCHS     = args.warmup_epochs
    cfg.FREEZE_CNN_EPOCHS = args.freeze_cnn_epochs
    cfg.PATIENCE          = args.patience
    cfg.PHASE_A_EPOCHS    = args.phase_a_epochs
    cfg.USE_AMP           = not args.no_amp
    cfg.MIN_TRACK_PURITY  = args.min_track_purity
    cfg.MIN_WINDOW_PURITY = args.min_window_purity

    if not os.path.isdir(cfg.DATA_ROOT):
        print(f"\n[ERROR] {cfg.DATA_ROOT} not found\n"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    loaders = build_dataloaders(cfg)
    if not loaders: print("[ERROR] No loaders"); sys.exit(1)

    model = EfficientNetB0_BiLSTM(
        lstm_hidden=cfg.LSTM_HIDDEN, lstm_layers=cfg.LSTM_LAYERS,
        lstm_drop=cfg.LSTM_DROPOUT, cnn_drop=cfg.CNN_DROPOUT)
    print(f"\n  Params: {sum(p.numel() for p in model.parameters()):,}  "
          f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer = Trainer(model, loaders, cfg, device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        trainer.load_checkpoint(args.checkpoint)

    if not args.test_only:
        trainer.train()

    tm = trainer.evaluate("test")
    generate_plots(cfg, tm)
    print(f"\n✓ Done  →  {cfg.RESULTS_DIR}/")


if __name__ == "__main__":
    main()