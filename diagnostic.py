import os, math
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

DATA_ROOT = "/mnt/projects/sutravek_project/Ali_belhrak/SI_Project/ALFIdatasetFinal/Data&Annotations"

SEQ_LEN           = 8
TRAIN_RATIO       = 0.70
VAL_RATIO         = 0.15
SEED              = 42

TASK2_CLASSES = {
    "earlymitosis": 0,
    "latemitosis":  1,
    "celldeath":    2,
    "multipolar":   3,
}

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD ANNOTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_annotations(seq_dir):
    name = os.path.basename(seq_dir)
    for fname in [
        f"{name}_PhenoTruth.csv",
        f"{name}_T2DTLTruth.csv",
        f"{name}_PhenotypeTruth.csv",
        f"{name}_DTLTruth.csv",
    ]:
        p = os.path.join(seq_dir, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip()
            df["Class"] = df["Class"].str.strip().str.lower().str.replace(" ", "")
            filtered = df[df["Class"].isin(TASK2_CLASSES)].reset_index(drop=True)
            if not filtered.empty:
                return filtered
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD ALL SAMPLES (center-frame labeling, same as your main script)
# ─────────────────────────────────────────────────────────────────────────────

def build_all_samples(data_root, seq_len):
    t2_seqs = [
        e for e in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, e, "Images"))
        and (e.upper().startswith("CD") or e.upper().startswith("TP"))
    ]

    all_samples = []
    for seq_name in t2_seqs:
        seq_dir   = os.path.join(data_root, seq_name)
        image_dir = os.path.join(seq_dir, "Images")
        df        = load_annotations(seq_dir)
        if df.empty or not os.path.isdir(image_dir):
            continue

        for cell_id, cdf in df.groupby("ID"):
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
            half         = seq_len // 2
            seen_windows = set()

            for s in range(n):
                center_class = class_labels[s]
                center_label = TASK2_CLASSES.get(center_class, -1)
                if center_label == -1:
                    continue
                if center_label == 0 and s % 3 != 0:
                    continue

                start = max(0, s - half)
                end   = start + seq_len
                if end > n:
                    start = max(0, n - seq_len)
                    end   = n

                fp_win = frame_paths[start:end]
                bb_win = bboxes[start:end]

                if len(fp_win) < seq_len:
                    pad    = seq_len - len(fp_win)
                    fp_win = list(fp_win) + [fp_win[-1]] * pad
                    bb_win = list(bb_win) + [bb_win[-1]] * pad
                else:
                    fp_win = list(fp_win)
                    bb_win = list(bb_win)

                win_key = (fp_win[0], fp_win[-1], center_label)
                if win_key in seen_windows:
                    continue
                seen_windows.add(win_key)

                all_samples.append(dict(
                    frames=fp_win,
                    bboxes=bb_win,
                    label=center_label,
                    cell_id=cell_id,
                    seq_name=seq_name,
                ))

    return all_samples


# ─────────────────────────────────────────────────────────────────────────────
#  SPLIT  (same logic as your build_dataloaders)
# ─────────────────────────────────────────────────────────────────────────────

def split_samples(all_samples, train_ratio, val_ratio, seed):
    labels = np.array([s["label"]   for s in all_samples])
    groups = np.array([f"{s['seq_name']}_{s['cell_id']}" for s in all_samples])
    rng    = np.random.default_rng(seed)

    train_idx_list, val_idx_list, test_idx_list = [], [], []
    test_ratio = 1.0 - train_ratio - val_ratio

    for cls in sorted(set(labels.tolist())):
        cls_mask   = np.where(labels == cls)[0]
        cls_groups = groups[cls_mask]

        unique_grps = np.array(sorted(set(cls_groups)))
        rng.shuffle(unique_grps)
        n = len(unique_grps)

        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        if n - n_test - n_val < 1 and n >= 3:
            n_val = max(1, n - n_test - 1)

        test_grps  = set(unique_grps[:n_test])
        val_grps   = set(unique_grps[n_test: n_test + n_val])

        for i in cls_mask:
            g = groups[i]
            if   g in test_grps: test_idx_list.append(i)
            elif g in val_grps:  val_idx_list.append(i)
            else:                train_idx_list.append(i)

    return (np.array(train_idx_list),
            np.array(val_idx_list),
            np.array(test_idx_list))


# ─────────────────────────────────────────────────────────────────────────────
#  LEAKAGE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_leakage(all_samples, train_idx, val_idx, test_idx):
    NAMES = {0:"earlymitosis", 1:"latemitosis", 2:"celldeath", 3:"multipolar"}

    print("\n" + "═"*65)
    print("  LEAKAGE CHECK")
    print("═"*65)

    # ── Sample counts per split ───────────────────────────────────────────
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        counts = Counter(all_samples[i]["label"] for i in idx)
        cnt_str = "  ".join(f"{NAMES[k]}={v}" for k, v in sorted(counts.items()))
        print(f"  {name:5s}  {len(idx):5d} samples  |  {cnt_str}")

    # ── Frame-level overlap ───────────────────────────────────────────────
    print()
    train_frames = set(f for i in train_idx for f in all_samples[i]["frames"])
    val_frames   = set(f for i in val_idx   for f in all_samples[i]["frames"])
    test_frames  = set(f for i in test_idx  for f in all_samples[i]["frames"])

    tr_te = train_frames & test_frames
    tr_vl = train_frames & val_frames
    vl_te = val_frames   & test_frames

    print(f"  Train frames : {len(train_frames)}")
    print(f"  Val frames   : {len(val_frames)}")
    print(f"  Test frames  : {len(test_frames)}")
    print(f"  Train ∩ Test : {len(tr_te):5d}  {'⚠  FRAME LEAKAGE' if tr_te else '✓ clean'}")
    print(f"  Train ∩ Val  : {len(tr_vl):5d}  {'⚠  FRAME LEAKAGE' if tr_vl else '✓ clean'}")
    print(f"  Val   ∩ Test : {len(vl_te):5d}  {'⚠  FRAME LEAKAGE' if vl_te else '✓ clean'}")

    # ── Cell-level overlap (the critical one) ─────────────────────────────
    print()
    train_cells = set(f"{all_samples[i]['seq_name']}_{all_samples[i]['cell_id']}" for i in train_idx)
    val_cells   = set(f"{all_samples[i]['seq_name']}_{all_samples[i]['cell_id']}" for i in val_idx)
    test_cells  = set(f"{all_samples[i]['seq_name']}_{all_samples[i]['cell_id']}" for i in test_idx)

    cell_tr_te = train_cells & test_cells
    cell_tr_vl = train_cells & val_cells
    cell_vl_te = val_cells   & test_cells

    print(f"  Train cells  : {len(train_cells)}")
    print(f"  Val cells    : {len(val_cells)}")
    print(f"  Test cells   : {len(test_cells)}")
    print(f"  Train ∩ Test : {len(cell_tr_te):5d}  {'⚠  CELL LEAKAGE — results are inflated' if cell_tr_te else '✓ clean'}")
    print(f"  Train ∩ Val  : {len(cell_tr_vl):5d}  {'⚠  CELL LEAKAGE' if cell_tr_vl else '✓ clean'}")
    print(f"  Val   ∩ Test : {len(cell_vl_te):5d}  {'⚠  CELL LEAKAGE' if cell_vl_te else '✓ clean'}")

    if cell_tr_te:
        print(f"\n  Leaked cells (first 10): {sorted(cell_tr_te)[:10]}")

    # ── Per-class cell distribution ───────────────────────────────────────
    print("\n  ── Per-class cell counts across splits ──")
    print(f"  {'Class':<15} {'Train cells':>12} {'Val cells':>10} {'Test cells':>10}")
    for cls in sorted(NAMES):
        tr = len(set(f"{all_samples[i]['seq_name']}_{all_samples[i]['cell_id']}"
                     for i in train_idx if all_samples[i]["label"] == cls))
        vl = len(set(f"{all_samples[i]['seq_name']}_{all_samples[i]['cell_id']}"
                     for i in val_idx   if all_samples[i]["label"] == cls))
        te = len(set(f"{all_samples[i]['seq_name']}_{all_samples[i]['cell_id']}"
                     for i in test_idx  if all_samples[i]["label"] == cls))
        print(f"  {NAMES[cls]:<15} {tr:>12} {vl:>10} {te:>10}")

    print("\n" + "═"*65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Building samples from {DATA_ROOT} ...")
    all_samples = build_all_samples(DATA_ROOT, SEQ_LEN)
    print(f"Total samples: {len(all_samples)}")

    train_idx, val_idx, test_idx = split_samples(
        all_samples, TRAIN_RATIO, VAL_RATIO, SEED
    )

    check_leakage(all_samples, train_idx, val_idx, test_idx)