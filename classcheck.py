import os
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG & DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATA_ROOT = (
    "/mnt/projects/sutravek_project/Ali_belhrak/"
    "SI_Project/ALFIdatasetFinal/Data&Annotations"
)

SEQ_FAMILIES = {
    "CD": ["CD01","CD02","CD03","CD04","CD05","CD06","CD07","CD08","CD09"],
    "MI": ["MI01","MI02","MI03","MI04","MI05","MI06","MI07","MI08"],
    "TP": ["TP01","TP02","TP03","TP04","TP05","TP06","TP07","TP08",
           "TP09","TP10","TP11","TP12"],
}

# Task 1: Generally Interphase vs Mitosis
TASK1_CSV_KEYS = ["dtltruth", "task1", "binary", "cellcycle"]
# Task 2: Generally phenotypes/subclasses
TASK2_CSV_KEYS = ["phenotype", "task2", "subclass", "pheno", "final"]

CLASS_COLUMN_CANDIDATES = [
    "Class", "class", "Label", "label",
    "Phenotype", "phenotype", "Category", "category",
]

# ─────────────────────────────────────────────────────────────────────────────
#  CORE DISCOVERY & SCANNING logic
# ─────────────────────────────────────────────────────────────────────────────

def find_sequences(data_root: str) -> List[str]:
    if not os.path.exists(data_root):
        print(f"[ERROR] Path does not exist: {data_root}")
        return []
    return [
        os.path.join(data_root, e)
        for e in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, e))
    ]

def find_class_column(df: pd.DataFrame) -> str | None:
    df.columns = df.columns.str.strip()
    for c in CLASS_COLUMN_CANDIDATES:
        if c in df.columns:
            return c
    return None

def scan_csv(path: str) -> pd.Series:
    try:
        df = pd.read_csv(path)
        col = find_class_column(df)
        if col is None:
            return pd.Series(dtype=int)
        # Clean labels: strip whitespace and convert to lowercase for consistency
        return df[col].astype(str).str.strip().str.lower().str.replace(" ", "_").value_counts()
    except Exception as e:
        print(f"    [ERROR] Reading {os.path.basename(path)}: {e}")
        return pd.Series(dtype=int)

def csv_task(filename: str) -> str | None:
    name = filename.lower()
    if not name.endswith(".csv"): return None
    if any(k in name for k in TASK1_CSV_KEYS): return "task1"
    if any(k in name for k in TASK2_CSV_KEYS): return "task2"
    return None

def scan_seq_dir(seq_dir: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    t1, t2 = defaultdict(int), defaultdict(int)
    try:
        files = sorted(os.listdir(seq_dir))
    except Exception:
        return dict(t1), dict(t2)

    for fname in files:
        task = csv_task(fname)
        if task is None: continue
        
        counts = scan_csv(os.path.join(seq_dir, fname))
        target_dict = t1 if task == "task1" else t2
        
        for cls, n in counts.items():
            target_dict[cls] += int(n)

    return dict(t1), dict(t2)

# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bar(count: int, total: int, width: int = 25) -> str:
    if total <= 0: return "░" * width
    filled = int(width * count / total)
    return "█" * filled + "░" * (width - filled)

def print_table(data_dict: Dict[str, int], title: str):
    print(f"\n--- {title} ---")
    total = sum(data_dict.values())
    if total == 0:
        print("    (No data found)")
        return
    
    # Sort by count descending
    for cls in sorted(data_dict, key=data_dict.get, reverse=True):
        n = data_dict[cls]
        pct = (n / total) * 100
        print(f"    {cls:<25} {n:>8,} {pct:>6.1f}%  {bar(n, total)}")
    print(f"    {'TOTAL':<25} {total:>8,}")

# ─────────────────────────────────────────────────────────────────────────────
#  SPLITTING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def get_splits(seq_names: List[str], tr=0.7, vl=0.15, seed=42):
    rng = np.random.default_rng(seed)
    train, val, test = set(), set(), set()
    
    # Stratify by family (CD, MI, TP)
    families = defaultdict(list)
    for s in seq_names:
        found_fam = "OTHER"
        for f_prefix in SEQ_FAMILIES:
            if s in SEQ_FAMILIES[f_prefix]:
                found_fam = f_prefix
                break
        families[found_fam].append(s)

    for f in families:
        members = families[f]
        rng.shuffle(members)
        n = len(members)
        n_tr = max(1, int(n * tr))
        n_vl = max(1, int(n * vl))
        
        train.update(members[:n_tr])
        val.update(members[n_tr : n_tr + n_vl])
        test.update(members[n_tr + n_vl :])
        
    return train, val, test

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    seq_paths = find_sequences(args.data_root)
    all_seqs = [os.path.basename(p) for p in seq_paths]
    
    if not all_seqs:
        print("No folders found. Check your path.")
        return

    g_t1, g_t2 = defaultdict(int), defaultdict(int)
    
    print(f"\nALFI DATASET SCANNER")
    print(f"Path: {args.data_root}")
    print(f"Sequences Found: {len(all_seqs)}")
    print("=" * 60)

    # 1. Grand Totals
    for s in all_seqs:
        t1, t2 = scan_seq_dir(os.path.join(args.data_root, s))
        for k, v in t1.items(): g_t1[k] += v
        for k, v in t2.items(): g_t2[k] += v

    print_table(g_t1, "GRAND TOTALS: TASK 1 (Binary/Cycle)")
    print_table(g_t2, "GRAND TOTALS: TASK 2 (Phenotypes)")

    # 2. Split Analysis
    tr_s, vl_s, ts_s = get_splits(all_seqs)
    
    for split_name, subset in [("TRAIN", tr_s), ("VAL", vl_s), ("TEST", ts_s)]:
        st1, st2 = defaultdict(int), defaultdict(int)
        for s in subset:
            t1, t2 = scan_seq_dir(os.path.join(args.data_root, s))
            for k, v in t1.items(): st1[k] += v
            for k, v in t2.items(): st2[k] += v
        
        print(f"\n{'='*15} SPLIT: {split_name} ({len(subset)} seqs) {'='*15}")
        print_table(st1, "Task 1 Distribution")
        print_table(st2, "Task 2 Distribution")

    print(f"\nTEST SEQS: {sorted(list(ts_s))}")

if __name__ == "__main__":
    main()