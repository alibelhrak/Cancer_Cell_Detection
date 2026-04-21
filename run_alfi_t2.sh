#!/bin/bash
#SBATCH --job-name=alfi_b0_task2
#SBATCH --output=effnetb0_task2_%j.out
#SBATCH --error=effnetb0_task2_%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --time=2-12:00:00

set -euo pipefail

echo "=================================================="
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Job Name      : $SLURM_JOB_NAME"
echo "  Node          : $(hostname)"
echo "  Start time    : $(date)"
echo "=================================================="

module purge || true
module load cuda/12.8.1

echo "CUDA:"
nvcc --version | head -1 || true

# Activate environment
source /mnt/projects/sutravek_project/Ali_belhrak/EfficientNetB7/Grad_Cam/next_step_venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Python : $(python --version)"
echo "Venv   : $VIRTUAL_ENV"

echo ""
echo "── GPU Status ────────────────────────────────────"
nvidia-smi || true
echo ""

# ── Sanity checks ─────────────────────────────────
echo "── Sanity Checks ─────────────────────────────────"

DATA_ROOT="/mnt/projects/sutravek_project/Ali_belhrak/SI_Project/ALFIdatasetFinal/Data&Annotations"
SCRIPT="/mnt/projects/sutravek_project/Ali_belhrak/SI_Project/ALFIdatasetFinal/alfi_train_t2.py"

if [ -d "$DATA_ROOT" ]; then
    echo "  DATA_ROOT OK : $DATA_ROOT"
    echo "  Sequences    : $(ls "$DATA_ROOT" | grep -E '^(CD|TP)[0-9]+$' | tr '\n' ' ')"
else
    echo "  [ERROR] DATA_ROOT not found: $DATA_ROOT"
    exit 1
fi

if [ -f "$SCRIPT" ]; then
    echo "  Script OK    : $SCRIPT"
else
    echo "  [ERROR] Script not found: $SCRIPT"
    exit 1
fi

echo ""

# PyTorch check
python - <<'PYEOF'
import torch, sys
print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name        :", torch.cuda.get_device_name(0))
    print("VRAM            :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
else:
    print("[WARNING] CUDA not available")
    sys.exit(1)
PYEOF

echo ""
echo "── Starting Training ─────────────────────────────"

NUM_WORKERS=4

echo "  Architecture  : EfficientNetB0 + BiLSTM + Attention"
echo "  Task          : Task 2 — Phenotype Classification (4 classes)"
echo "  Workers       : $NUM_WORKERS"
echo "Time: $(date)"
echo ""

python "$SCRIPT" \
    --data_root          "$DATA_ROOT" \
    --save_dir           /mnt/projects/sutravek_project/Ali_belhrak/SI_Project/ALFIdatasetFinal/newcheckpoints_task2 \
    --results_dir        /mnt/projects/sutravek_project/Ali_belhrak/SI_Project/ALFIdatasetFinal/newresults_task2 \
    --seq_len            8 \
    --img_size           224 \
    --batch_size         8 \
    --num_workers        $NUM_WORKERS \
    --n_accum            4 \
    --epochs             35 \
    --lr                 1e-4 \
    --weight_decay       1e-2 \
    --warmup_epochs      3 \
    --freeze_cnn_epochs  3 \
    --phase_a_epochs     8 \
    --patience           8 \
    --min_track_purity   0.60 \
    --min_window_purity  0.60

echo ""
echo "── Training Finished ─────────────────────────────"
echo "End time: $(date)"

deactivate