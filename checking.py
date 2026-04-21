python3 - <<'EOF'
import os, pandas as pd

DATA_ROOT = "/mnt/projects/sutravek_project/Ali_belhrak/SI_Project/ALFIdatasetFinal/Data&Annotations"
TASK2_CLASSES = {"earlymitosis","latemitosis","celldeath","multipolar"}

seqs = [e for e in sorted(os.listdir(DATA_ROOT))
        if os.path.isdir(os.path.join(DATA_ROOT, e, "Images"))
        and (e.upper().startswith("CD") or e.upper().startswith("TP"))]

print(f"Found {len(seqs)} sequences: {seqs}\n")

for seq in seqs:
    seq_dir = os.path.join(DATA_ROOT, seq)
    csvs = [f for f in os.listdir(seq_dir) if f.endswith(".csv")]
    print(f"── {seq}: CSVs = {csvs}")
    for csv in csvs:
        df = pd.read_csv(os.path.join(seq_dir, csv))
        df.columns = df.columns.str.strip()
        if "Class" in df.columns:
            raw   = df["Class"].unique().tolist()
            normd = df["Class"].str.strip().str.lower().str.replace(" ","").unique().tolist()
            match = [c for c in normd if c in TASK2_CLASSES]
            print(f"   {csv}")
            print(f"     raw classes  : {raw}")
            print(f"     normalized   : {normd}")
            print(f"     matched T2   : {match}")
        else:
            print(f"   {csv} — NO 'Class' column! Columns: {df.columns.tolist()}")
    print()
EOF