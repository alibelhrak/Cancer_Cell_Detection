#!/bin/bash
#SBATCH --job-name=alfi_b0
#SBATCH --output=effnetb0_%j.out
#SBATCH --error=effnetb0_%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=2-12:00:00

set -e

echo "=================================================="
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Job Name      : $SLURM_JOB_NAME"
echo "  Node          : $(hostname)"
echo "  Start time    : $(date)"
echo "  Working dir   : $(pwd)"
echo "  CPUs          : $SLURM_CPUS_PER_TASK"
echo "  Memory        : $SLURM_MEM_PER_NODE MB"
echo "=================================================="

module purge
module load cuda/12.8.1

nvcc --version | head -1 || true

source /mnt/projects/sutravek_project/Ali_belhrak/EfficientNetB7/Grad_Cam/next_step_venv/bin/activate

echo "Python : $(python --version)"
echo "Venv   : $VIRTUAL_ENV"

echo ""
echo "── GPU Status ────────────────────────────────────"
nvidia-smi || true
echo ""

python - <<EOF
import torch
print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name        :", torch.cuda.get_device_name(0))
    print("VRAM            :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
    print("CUDA version    :", torch.version.cuda)
else:
    print("WARNING: CUDA NOT AVAILABLE")
EOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

echo ""
echo "── Starting Training ─────────────────────────────"
echo "Time: $(date)"
echo ""

python /mnt/projects/sutravek_project/Ali_belhrak/SI_Project/ALFIdatasetFinal/Alfi.py \
    --epochs        30 \
    --batch_size    8 \
    --seq_len       8 \
    --num_workers   4 \
    --patience      10 \
    --save_dir      /mnt/projects/sutravek_project/Ali_belhrak/SI_Project/ALFIdatasetFinal/checkpoints1 \
    --results_dir   /mnt/projects/sutravek_project/Ali_belhrak/SI_Project/ALFIdatasetFinal/results1

echo ""
echo "── Training Finished ─────────────────────────────"
echo "End time: $(date)"

deactivate