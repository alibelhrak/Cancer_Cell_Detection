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
    SAVE_DIR    = "./newcheckpoints_task2"
    RESULTS_DIR = "./newresults_task2"

    SEQ_LEN     = 8
    IMG_SIZE    = 224
    BATCH_SIZE  = 8
    NUM_WORKERS = 4
    CROP_CELLS  = True
    N_ACCUM     = 4

    T2_SPLIT_TRAIN = 0.70
    T2_SPLIT_VAL   = 0.15
    T2_SPLIT_TEST  = 0.15

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

    TASK2_LOSS_WEIGHT_A = 1.0
    TASK2_LOSS_WEIGHT_B = 1.0

    PATIENCE  = 8
    USE_AMP   = True

    MIN_TRACK_PURITY  = 0.5
    MIN_WINDOW_PURITY = 0.5


# ── Task 2: 4-class phenotype labels ─────────────────────────────────────────
# Map the exact strings that appear in the CSV "Class" column
TASK2_CLASSES = {
    "earlymitosis": 0,
    "latemitosis":  1,
    "celldeath":    2,
    "multipolar":   3,
}
TASK2_NAMES = {v: k for k, v in TASK2_CLASSES.items()}

# Sequences that contain Task-2 annotations (CD* and TP*)
SEQ_FAMILIES = {
    "CD": ["CD01","CD02","CD03","CD04","CD05","CD06","CD07","CD08","CD09"],
    "TP": ["TP01","TP02","TP03","TP04","TP05","TP06","TP07",
           "TP08","TP09","TP10","TP11","TP12"],
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
#  CELL-FAMILY SPLIT (Task 2)
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
#  ANNOTATION LOADING  — Task 2 uses *_T2DTLTruth.csv (or similar)
# ─────────────────────────────────────────────────────────────────────────────

def load_annotations(seq_dir: str) -> pd.DataFrame:
    """
    Try several candidate filenames for the Task-2 annotation CSV.
    The ALFI dataset uses _DTLTruth.csv for Task-1 sequences and
    _T2DTLTruth.csv (or _PhenotypeTruth.csv) for Task-2 sequences.
    We fall back to the plain _DTLTruth.csv if nothing else is found,
    and then filter to only Task-2 class names.
    """
    name = os.path.basename(seq_dir)
    candidates = [
        os.path.join(seq_dir, f"{name}_PhenoTruth.csv"),     # actual ALFI Task-2 filename
        os.path.join(seq_dir, f"{name}_T2DTLTruth.csv"),
        os.path.join(seq_dir, f"{name}_PhenotypeTruth.csv"),
        os.path.join(seq_dir, f"{name}_DTLTruth.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip()
            # Normalise class strings: lowercase, remove spaces
            df["Class"] = df["Class"].str.strip().str.lower().str.replace(" ", "")
            filtered = df[df["Class"].isin(TASK2_CLASSES)].reset_index(drop=True)
            if not filtered.empty:
                return filtered
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  TRACK BUILDER  — identical logic to Task-1 version, uses TASK2_CLASSES
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

    label_map = TASK2_CLASSES

    if use_family_split and split != "all":
        tr_ids, vl_ids, te_ids = cell_family_split(
            df, label_map, train_ratio, val_ratio, seed
        )
        active_ids = {
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

        frame_nums   = cdf["ImNo"].tolist()
        frame_paths  = [
            os.path.join(image_dir, f"I_{seq_name}_{fn:04d}.png")
            for fn in frame_nums
        ]
        bboxes       = list(zip(cdf["xmin"], cdf["ymin"],
                                cdf["width"], cdf["height"]))
        class_labels = cdf["Class"].tolist()

        # Filter to existing images only
        valid = [
            (p, b, c)
            for p, b, c in zip(frame_paths, bboxes, class_labels)
            if os.path.exists(p)
        ]
        if len(valid) < 2:
            continue
        frame_paths, bboxes, class_labels = zip(*valid)
        frame_paths  = list(frame_paths)
        bboxes       = list(bboxes)
        class_labels = list(class_labels)

        n    = len(frame_paths)
        half = seq_len // 2

        # ── CENTER-FRAME LABELING ─────────────────────────────────────────
        # Each window is labeled by its center frame's class.
        # This correctly captures transient events like multipolar division
        # which never dominate a full track but appear for 2-4 frames.
        #
        # For earlymitosis (dominant class) we use a coarser stride to avoid
        # generating too many redundant windows.
        # For all minority classes (latemitosis, celldeath, multipolar) we
        # slide every frame so we don't miss any rare event.

        seen_windows = set()  # deduplicate identical windows

        for s in range(n):
            center_class = class_labels[s]
            center_label = label_map.get(center_class, -1)
            if center_label == -1:
                continue

            # Coarse stride for earlymitosis — it has 5784 annotations,
            # we don't need every possible window
            if center_label == 0 and s % 5 != 0:
                continue

            # Build window centered on frame s
            start = max(0, s - half)
            end   = start + seq_len
            if end > n:
                start = max(0, n - seq_len)
                end   = n

            fp_win = frame_paths[start:end]
            bb_win = bboxes[start:end]

            # Pad if track shorter than seq_len
            if len(fp_win) < seq_len:
                pad    = seq_len - len(fp_win)
                fp_win = list(fp_win) + [fp_win[-1]] * pad
                bb_win = list(bb_win) + [bb_win[-1]] * pad
            else:
                fp_win = list(fp_win)
                bb_win = list(bb_win)

            # Deduplicate — same start frame + same label = same window
            win_key = (fp_win[0], fp_win[-1], center_label)
            if win_key in seen_windows:
                continue
            seen_windows.add(win_key)

            samples.append(dict(
                frames=fp_win, bboxes=bb_win,
                label=center_label,
                cell_id=cell_id,
                seq_name=seq_name,
            ))

        # TEMPORAL AUGMENTATION for minority classes
        # Collect the base windows we just added for this cell,
        # then augment minority classes only.
        if center_label in (1, 2, 3):  # latemitosis, celldeath, multipolar
            cell_samples = [
                s for s in samples
                if s["cell_id"] == cell_id
                and s["seq_name"] == seq_name
                and s["label"] == center_label
            ]

            N_AUG = {1: 2, 2: 2, 3: 5}  # multipolar gets the most copies
            n_aug = N_AUG.get(center_label, 1)

            for w in cell_samples:
                fp = w["frames"]
                bb = w["bboxes"]
                aug_count = 0

                # Aug 1: reverse sequence
                if aug_count < n_aug:
                    samples.append(dict(
                        frames=list(reversed(fp)),
                        bboxes=list(reversed(bb)),
                        label=center_label,
                        cell_id=cell_id,
                        seq_name=seq_name,
                        augmented=True,
                    ))
                    aug_count += 1

                # Aug 2: temporal subsampling
                if aug_count < n_aug and len(fp) >= 4:
                    fp2 = fp[::2]; bb2 = bb[::2]
                    pad = seq_len - len(fp2)
                    if pad > 0:
                        fp2 = fp2 + [fp2[-1]] * pad
                        bb2 = bb2 + [bb2[-1]] * pad
                    samples.append(dict(
                        frames=fp2[:seq_len],
                        bboxes=bb2[:seq_len],
                        label=center_label,
                        cell_id=cell_id,
                        seq_name=seq_name,
                        augmented=True,
                    ))
                    aug_count += 1

                # Aug 3+: repeat reverse to hit target count (multipolar only)
                while aug_count < n_aug:
                    samples.append(dict(
                        frames=list(reversed(fp)),
                        bboxes=list(reversed(bb)),
                        label=center_label,
                        cell_id=cell_id,
                        seq_name=seq_name,
                        augmented=True,
                    ))
                    aug_count += 1

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
        dist_str = "  ".join(f"{TASK2_NAMES[k]}:{v}" for k,v in sorted(dist.items()))
        print(f"  [Task 2 | {split:5s}]  {len(self.samples):5d} samples  |  {dist_str}")

    def _load_frame(self, path: str, bbox=None) -> Image.Image:
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
    sw = [1.0 / lc[s["label"]] for s in ds.samples]  # plain inverse freq, no boost
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

    # Collect CD* and TP* sequence directories
    t2_seqs = [
        e for e in sorted(os.listdir(cfg.DATA_ROOT))
        if os.path.isdir(os.path.join(cfg.DATA_ROOT, e, "Images"))
        and (e.upper().startswith("CD") or e.upper().startswith("TP"))
    ]
    if not t2_seqs:
        print("[WARNING] No CD/TP sequences found for Task 2 — check DATA_ROOT")
        return loaders

    print(f"\n── Task 2: {len(t2_seqs)} sequences ──")

    # ── Step 1: collect every (seq_name, cell_id, label) from CSVs ───────
    # We do this BEFORE building windows so we can split at the cell level.
    from collections import defaultdict
    cell_label_map: Dict[str, set] = defaultdict(set)  # cell_key → set of labels

    for seq_name in t2_seqs:
        seq_dir = os.path.join(cfg.DATA_ROOT, seq_name)
        df      = load_annotations(seq_dir)
        if df.empty:
            continue
        for cell_id, cdf in df.groupby("ID"):
            cdf = cdf[cdf["Class"].isin(TASK2_CLASSES)]
            if cdf.empty:
                continue
            cell_key = f"{seq_name}_{cell_id}"
            for cls in cdf["Class"].unique():
                cell_label_map[cell_key].add(TASK2_CLASSES[cls])

    if not cell_label_map:
        print("[ERROR] No annotated cells found"); return loaders

    # ── Step 2: assign each cell to a class bin for stratified splitting ──
    # A cell can carry multiple labels (center-frame labeling).
    # We assign it to its rarest label bin so minority classes are
    # represented in every split.
    RARITY = {0: 0, 1: 1, 2: 2, 3: 3}   # higher = rarer
    class_cells: Dict[int, list] = defaultdict(list)
    for cell_key, labels in cell_label_map.items():
        rarest_label = max(labels, key=lambda l: RARITY[l])
        class_cells[rarest_label].append(cell_key)

    # ── Step 3: stratified split at the cell level ────────────────────────
    rng = np.random.default_rng(cfg.SPLIT_SEED)
    train_cells: set = set()
    val_cells:   set = set()
    test_cells:  set = set()

    print("\n  Cell-level stratified split:")
    for cls in sorted(class_cells):
        cells = class_cells[cls]
        arr   = np.array(sorted(cells))
        rng.shuffle(arr)
        n      = len(arr)
        n_test = max(1, int(n * cfg.T2_SPLIT_TEST))
        n_val  = max(1, int(n * cfg.T2_SPLIT_VAL))
        if n - n_test - n_val < 1 and n >= 3:
            n_val = max(1, n - n_test - 1)
        elif n < 3:
            # very few cells — put at least one in test, rest in train
            n_test = 1; n_val = 0

        test_cells.update(arr[:n_test])
        val_cells.update(arr[n_test: n_test + n_val])
        train_cells.update(arr[n_test + n_val:])

        print(f"    cls={TASK2_NAMES[cls]:<15}  "
              f"total={n:3d}  "
              f"train={n - n_test - n_val:3d}  "
              f"val={n_val:3d}  "
              f"test={n_test:3d}")

    # Sanity — no cell should be in two splits
    assert not (train_cells & val_cells),  "train/val cell overlap!"
    assert not (train_cells & test_cells), "train/test cell overlap!"
    assert not (val_cells   & test_cells), "val/test cell overlap!"
    print(f"\n  ✓ Zero cell overlap confirmed")
    print(f"  train={len(train_cells)} cells  "
          f"val={len(val_cells)} cells  "
          f"test={len(test_cells)} cells")

    # ── Step 4: build windows per split using only the assigned cells ─────
    def build_split_samples(allowed_cells: set, split_name: str) -> List[Dict]:
        samples = []
        for seq_name in t2_seqs:
            seq_dir   = os.path.join(cfg.DATA_ROOT, seq_name)
            image_dir = os.path.join(seq_dir, "Images")
            df        = load_annotations(seq_dir)
            if df.empty or not os.path.isdir(image_dir):
                continue

            for cell_id, cdf in df.groupby("ID"):
                cell_key = f"{seq_name}_{cell_id}"
                if cell_key not in allowed_cells:
                    continue

                cdf = cdf.sort_values("ImNo").reset_index(drop=True)
                cdf = cdf[cdf["Class"].isin(TASK2_CLASSES)]
                if cdf.empty:
                    continue

                frame_nums   = cdf["ImNo"].tolist()
                frame_paths  = [
                    os.path.join(image_dir, f"I_{seq_name}_{fn:04d}.png")
                    for fn in frame_nums
                ]
                bboxes       = list(zip(cdf["xmin"], cdf["ymin"],
                                        cdf["width"], cdf["height"]))
                class_labels = cdf["Class"].tolist()

                # Filter to existing images only
                valid = [
                    (p, b, c)
                    for p, b, c in zip(frame_paths, bboxes, class_labels)
                    if os.path.exists(p)
                ]
                if len(valid) < 2:
                    continue

                frame_paths, bboxes, class_labels = zip(*valid)
                frame_paths  = list(frame_paths)
                bboxes       = list(bboxes)
                class_labels = list(class_labels)
                n            = len(frame_paths)
                half         = cfg.SEQ_LEN // 2
                seen_windows = set()

                # ── Center-frame labeling ─────────────────────────────────
                for s in range(n):
                    center_class = class_labels[s]
                    center_label = TASK2_CLASSES.get(center_class, -1)
                    if center_label == -1:
                        continue

                    # Coarse stride for earlymitosis — plenty of data
                    if center_label == 0 and s % 3 != 0:
                        continue

                    # Build window centered on frame s
                    start = max(0, s - half)
                    end   = start + cfg.SEQ_LEN
                    if end > n:
                        start = max(0, n - cfg.SEQ_LEN)
                        end   = n

                    fp_win = frame_paths[start:end]
                    bb_win = bboxes[start:end]

                    # Pad if track shorter than seq_len
                    if len(fp_win) < cfg.SEQ_LEN:
                        pad    = cfg.SEQ_LEN - len(fp_win)
                        fp_win = list(fp_win) + [fp_win[-1]] * pad
                        bb_win = list(bb_win) + [bb_win[-1]] * pad
                    else:
                        fp_win = list(fp_win)
                        bb_win = list(bb_win)

                    win_key = (fp_win[0], fp_win[-1], center_label)
                    if win_key in seen_windows:
                        continue
                    seen_windows.add(win_key)

                    samples.append(dict(
                        frames=fp_win,
                        bboxes=bb_win,
                        label=center_label,
                        cell_id=cell_id,
                        seq_name=seq_name,
                    ))

        return samples

    print("\n  Building windows per split...")
    train_samples = build_split_samples(train_cells, "train")
    val_samples   = build_split_samples(val_cells,   "val")
    test_samples  = build_split_samples(test_cells,  "test")

    if not train_samples:
        print("[ERROR] No training samples found"); return loaders

    # ── Step 5: verify zero frame leakage ────────────────────────────────
    train_frame_set = set(f for s in train_samples for f in s["frames"])
    val_frame_set   = set(f for s in val_samples   for f in s["frames"])
    test_frame_set  = set(f for s in test_samples  for f in s["frames"])
    tr_te_overlap   = train_frame_set & test_frame_set
    tr_vl_overlap   = train_frame_set & val_frame_set
    print(f"\n  Frame overlap train∩test : {len(tr_te_overlap)}  "
          f"{'⚠ leakage' if tr_te_overlap else '✓ clean'}")
    print(f"  Frame overlap train∩val  : {len(tr_vl_overlap)}  "
          f"{'⚠ leakage' if tr_vl_overlap else '✓ clean'}")

    # ── Step 6: class weights from training labels ────────────────────────
    tr_labels = np.array([s["label"] for s in train_samples])
    classes   = sorted(set(tr_labels.tolist()))
    w         = compute_class_weight("balanced", classes=np.array(classes), y=tr_labels)
    w_sq = w      # square to amplify minority signal

    weight_tensor = torch.ones(len(TASK2_CLASSES), dtype=torch.float)
    for i, c in enumerate(classes):
        weight_tensor[c] = float(w_sq[i])

    print(f"\n  Task2 loss weights (balanced²): "
          f"{ {TASK2_NAMES[c]: f'{w_sq[i]:.3f}' for i, c in enumerate(classes)} }")

    loaders["task2_weights"] = weight_tensor

    # ── Step 7: build datasets and dataloaders ────────────────────────────
    loaders["task2"] = {}
    for split, samples in [("train", train_samples),
                            ("val",   val_samples),
                            ("test",  test_samples)]:
        ds = ALFIDatasetFromSamples(
            samples, split, cfg.IMG_SIZE, cfg.CROP_CELLS
        )
        loaders["task2"][split] = (
            _make_weighted_loader(ds, cfg) if split == "train"
            else _make_loader(ds, cfg)
        )

    return loaders
# ─────────────────────────────────────────────────────────────────────────────
#  MODEL  — same architecture, head output = 4 classes
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetB0_BiLSTM(nn.Module):
    def __init__(self, num_classes_t2=4,
                 lstm_hidden=256, lstm_layers=1, lstm_drop=0.5, cnn_drop=0.5):
        super().__init__()
        bb = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cnn_features = bb.features
        self.cnn_avgpool  = bb.avgpool
        self.cnn_drop     = nn.Dropout(cnn_drop)
        CNN_DIM = 1280

        self.lstm = nn.LSTM(CNN_DIM, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=lstm_drop if lstm_layers > 1 else 0.0)
        LSTM_DIM = lstm_hidden * 2

        self.attention = nn.Sequential(
            nn.Linear(LSTM_DIM, 128), nn.Tanh(), nn.Linear(128, 1))

        # 4-class head (Task 2)
        self.head_t2 = nn.Sequential(
            nn.LayerNorm(LSTM_DIM), nn.Dropout(0.40),
            nn.Linear(LSTM_DIM, 64), nn.ReLU(True),
            nn.Dropout(0.20), nn.Linear(64, num_classes_t2))

        self._init_weights()

    def _init_weights(self):
        for n, p in self.lstm.named_parameters():
            if   "weight_ih" in n: nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in n: nn.init.orthogonal_(p.data)
            elif "bias"      in n:
                p.data.fill_(0); k = p.size(0)
                p.data[k//4:k//2].fill_(1.0)
        for mod in [self.head_t2, self.attention]:
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
        feats       = self._cnn(x.view(B*T, C, H, W)).view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        attn_w      = torch.softmax(self.attention(lstm_out), dim=1)
        context     = (attn_w * lstm_out).sum(dim=1)
        return self.head_t2(context), attn_w.squeeze(-1)


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
        if ep < warmup: return (ep+1) / max(warmup, 1)
        prog = (ep-warmup) / max(total-warmup+1, 1)
        return max(0.5*(1+np.cos(np.pi*prog)), min_f)
    return optim.lr_scheduler.LambdaLR(opt, _lam)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────────────────────────────────────

_D = {"bg":"#0f1117","panel":"#1a1e2e","a1":"#4fc3f7","a2":"#f48fb1",
      "a3":"#a5d6a7","a4":"#ffcc02","text":"#e0e0e0","grid":"#2a2a3e"}


class Trainer:
    def __init__(self, model, loaders, cfg, device):
        self.model   = model.to(device)
        self.loaders = loaders
        self.cfg     = cfg
        self.device  = device
        self.use_amp = cfg.USE_AMP and device.type == "cuda"

        os.makedirs(cfg.SAVE_DIR,    exist_ok=True)
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

        t2w = loaders.get("task2_weights")
        self.crit_t2 = FocalLoss(2.0, None)

        cnn_p   = list(self.model.cnn_features.parameters())
        other_p = [p for p in model.parameters()
                   if not any(p is c for c in cnn_p)]
        self.optimizer = optim.AdamW(
            [{"params": cnn_p,   "lr": cfg.LR * 0.1},
             {"params": other_p, "lr": cfg.LR}],
            weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = build_scheduler(self.optimizer, cfg.WARMUP_EPOCHS, cfg.EPOCHS)

        self.plateau = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-7)

        self.scaler = GradScaler() if self.use_amp else None

        self.history = {k: [] for k in
            ["train_loss", "val_loss", "val_acc_t2", "train_acc_t2",
             "val_f1_t2",  "val_f1_t2_macro"]}
        self.best_f1   = -1.0
        self.pat_count = 0

        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 5))
        self.fig.suptitle("Live Training — Task 2 Phenotypes", fontsize=14)
        plt.show()

    def _update_plot(self):
        ep = range(1, len(self.history["train_loss"]) + 1)
        titles = ["Focal Loss", "Val Accuracy", "Val F1"]
        data = [
            [(self.history["train_loss"], "Train", _D["a1"]),
             (self.history["val_loss"],   "Val",   _D["a2"])],
            [(self.history["val_acc_t2"],   "Val",   _D["a1"]),
             (self.history["train_acc_t2"], "Train", _D["a2"])],
            [(self.history["val_f1_t2_macro"], "macro",    _D["a1"]),
             (self.history["val_f1_t2"],       "weighted", _D["a2"])],
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
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        out = os.path.join(self.cfg.RESULTS_DIR, "training_curves_live.png")
        self.fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=_D["bg"])

    def _get_w(self, ep):
        if ep <= self.cfg.PHASE_A_EPOCHS:
            return self.cfg.TASK2_LOSS_WEIGHT_A
        return self.cfg.TASK2_LOSS_WEIGHT_B

    def _forward(self, batch, w2):
        fr, lb = batch
        fr = fr.to(self.device, non_blocking=True)
        lb = lb.to(self.device, non_blocking=True)
        logits, _ = self.model(fr)
        loss  = w2 * self.crit_t2(logits, lb)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().tolist()
        labels = lb.cpu().tolist()
        return loss, (preds, labels)

    def _epoch(self, split, epoch=0):
        is_tr = split == "train"
        self.model.train() if is_tr else self.model.eval()

        loader = self.loaders.get("task2", {}).get(split)
        if loader is None:
            return {"loss": 0, "acc_t2": 0, "f1_t2": 0, "f1_t2_macro": 0,
                    "preds_t2": ([], [])}

        w2    = self._get_w(epoch)
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
                        loss, (preds, labels) = self._forward(batch, w2)
                else:
                    loss, (preds, labels) = self._forward(batch, w2)

                if is_tr:
                    loss_scaled = loss / accum
                    if self.scaler:
                        self.scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()

                    step_num += 1
                    if step_num % accum == 0 or step == len(loader) - 1:
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
        acc         = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        f1_weighted = f1_score(all_labels, all_preds, average="weighted",
                                zero_division=0) if all_labels else 0.0
        f1_macro    = f1_score(all_labels, all_preds, average="macro",
                                zero_division=0) if all_labels else 0.0

        return {
            "loss":        avg,
            "acc_t2":      acc,
            "f1_t2":       f1_weighted,
            "f1_t2_macro": f1_macro,
            "preds_t2":    (all_preds, all_labels),
        }

    def train(self):
        cfg = self.cfg
        print(f"\n{'═'*65}")
        print(f"  EfficientNetB0+BiLSTM  ALFI — Task 2 Only (4-class phenotypes)")
        print(f"  Device: {self.device}  AMP: {self.use_amp}")
        print(f"  Gradient accumulation: {cfg.N_ACCUM} steps (eff. batch={cfg.BATCH_SIZE*cfg.N_ACCUM})")
        print(f"  Early stopping: macro F1 (Task 2)")
        print(f"  Phase A ep1-{cfg.PHASE_A_EPOCHS}: T2×{cfg.TASK2_LOSS_WEIGHT_A}")
        print(f"  Phase B ep{cfg.PHASE_A_EPOCHS+1}+: T2×{cfg.TASK2_LOSS_WEIGHT_B}")
        print(f"{'═'*65}\n")

        self.model.freeze_cnn()
        print(f"[Phase 1] CNN frozen for {cfg.FREEZE_CNN_EPOCHS} epochs")

        for ep in range(1, cfg.EPOCHS + 1):
            t0 = time.time()
            if ep == cfg.FREEZE_CNN_EPOCHS + 1:
                self.model.unfreeze_cnn()
                print(f"\n[Phase 2] CNN unfrozen\n")

            lr  = self.optimizer.param_groups[1]["lr"]
            tr  = self._epoch("train", ep)
            val = self._epoch("val",   ep)
            self.scheduler.step()

            cf1 = val["f1_t2_macro"]
            if ep > cfg.WARMUP_EPOCHS:
                self.plateau.step(cf1)

            for k, v in [
                ("train_loss",      tr["loss"]),
                ("val_loss",        val["loss"]),
                ("val_acc_t2",      val["acc_t2"]),
                ("train_acc_t2",    tr["acc_t2"]),
                ("val_f1_t2",       val["f1_t2"]),
                ("val_f1_t2_macro", val["f1_t2_macro"]),
            ]:
                self.history[k].append(v)
            self._update_plot()
            print(f"Ep {ep:3d}/{cfg.EPOCHS}  lr={lr:.2e}  "
                  f"tr={tr['loss']:.4f}  val={val['loss']:.4f}  "
                  f"acc={val['acc_t2']:.3f}  "
                  f"f1={val['f1_t2_macro']:.3f}(macro)  "
                  f"f1={val['f1_t2']:.3f}(weighted)  "
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

        m      = self._epoch(split, self.cfg.EPOCHS)
        p2, l2 = m["preds_t2"]

        if l2:
            print("── Task 2 ──────────────────────────────────────────────")
            print(classification_report(
                l2, p2,
                labels=list(range(len(TASK2_CLASSES))),
                target_names=[TASK2_NAMES[i] for i in sorted(TASK2_NAMES)],
                zero_division=0))
            print(confusion_matrix(l2, p2,
                  labels=list(range(len(TASK2_CLASSES)))), "\n")
        return m

    def _save(self, tag, epoch, metrics):
        torch.save(
            {"epoch": epoch,
             "model_state_dict":     self.model.state_dict(),
             "optimizer_state_dict": self.optimizer.state_dict(),
             **{k: metrics[k] for k in
                ["loss", "acc_t2", "f1_t2", "f1_t2_macro"]}},
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
    ep  = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("EfficientNetB0+BiLSTM — Task 2 Phenotypes",
                 fontsize=14, color=_D["text"], fontweight="bold")
    c = [_D["a1"], _D["a2"], _D["a3"], _D["a4"]]

    axes[0].plot(ep, history["train_loss"], color=c[0], lw=2, label="Train")
    axes[0].plot(ep, history["val_loss"],   color=c[1], lw=2, ls="--", label="Val")
    axes[0].set_title("Focal Loss"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(ep, history["val_acc_t2"], color=c[0], lw=2, label="Task 2")
    axes[1].set_title("Val Accuracy"); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(ep, history["val_f1_t2_macro"], color=c[0], lw=2,       label="T2 macro")
    axes[2].plot(ep, history["val_f1_t2"],       color=c[1], lw=1.5, ls=":", label="T2 weighted")
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
    p2, l2 = tm.get("preds_t2", ([], []))
    if l2:
        plot_cm(l2, p2,
                [TASK2_NAMES[i] for i in sorted(TASK2_NAMES)],
                "Task 2 — Phenotype Classification",
                os.path.join(cfg.RESULTS_DIR, "cm_task2.png"),
                labels=list(range(len(TASK2_CLASSES))))
    save_metrics_csv(cfg, tm, split="test")


def save_metrics_csv(cfg, tm, split="test"):
    p2, l2 = tm.get("preds_t2", ([], []))
    if not l2:
        print("  [SKIP] No predictions to save.")
        return

    from sklearn.metrics import precision_score, recall_score

    class_names = [TASK2_NAMES[i] for i in sorted(TASK2_NAMES)]
    n_cls       = len(class_names)
    labels_list = list(range(n_cls))

    precision = precision_score(l2, p2, average=None, labels=labels_list, zero_division=0)
    recall    = recall_score(l2, p2,    average=None, labels=labels_list, zero_division=0)
    f1_per    = f1_score(l2, p2,        average=None, labels=labels_list, zero_division=0)

    acc         = accuracy_score(l2, p2)
    f1_macro    = f1_score(l2, p2, average="macro",    zero_division=0)
    f1_weighted = f1_score(l2, p2, average="weighted", zero_division=0)

    cm = confusion_matrix(l2, p2, labels=labels_list)

    rows = []
    for i, cname in enumerate(class_names):
        rows += [
            {"Metric": "Precision", "Class": cname, "Value": round(float(precision[i]), 4)},
            {"Metric": "Recall",    "Class": cname, "Value": round(float(recall[i]),    4)},
            {"Metric": "F1 Score",  "Class": cname, "Value": round(float(f1_per[i]),    4)},
        ]
    rows += [
        {"Metric": "Accuracy",    "Class": "Overall", "Value": round(acc,         4)},
        {"Metric": "F1 Macro",    "Class": "Overall", "Value": round(f1_macro,    4)},
        {"Metric": "F1 Weighted", "Class": "Overall", "Value": round(f1_weighted, 4)},
    ]
    # Per-class support counts
    for i, cname in enumerate(class_names):
        rows.append({"Metric": "Test Samples", "Class": cname,
                     "Value": int(cm[i].sum())})
    rows.append({"Metric": "Test Samples", "Class": "Total", "Value": int(len(l2))})

    df  = pd.DataFrame(rows)
    out = os.path.join(cfg.RESULTS_DIR, f"metrics_{split}.csv")
    df.to_csv(out, index=False)
    print(f"  Metrics → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p   = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    cfg  = Config()
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
        num_classes_t2=len(TASK2_CLASSES),
        lstm_hidden=cfg.LSTM_HIDDEN, lstm_layers=cfg.LSTM_LAYERS,
        lstm_drop=cfg.LSTM_DROPOUT,  cnn_drop=cfg.CNN_DROPOUT)
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