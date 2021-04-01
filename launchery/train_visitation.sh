#!/bin/bash
#SBATCH -J visitation-predictor
#SBATCH --output=/checkpoint/%u/jobs/job.%A_%a.out
#SBATCH --error=/checkpoint/%u/jobs/job.%A_%a.err
#SBATCH --gpus-per-task 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=learnfair
#SBATCH --time=6:00:00
#SBATCH --signal=USR1@300
#SBATCH --array=0-511%50
#SBATCH --open-mode=append

. /private/home/erikwijmans/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate nav-analysis-base


time_offset=$((${SLURM_ARRAY_TASK_ID} - ${SLURM_ARRAY_TASK_COUNT} / 2))
if (( ${time_offset} >= 0 )); then
    time_offset=$((${time_offset} + 1))
fi


module purge
module load cuda/10.1
module load cudnn/v7.6.5.32-cuda.10.1
module load NCCL/2.5.6-1-cuda.10.1
module load openmpi/4.0.1/gcc.7.4.0-git_patch#6654


export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

printenv | grep SLURM
set -x
srun -u \
    python -u -m nav_analysis.map_extraction.training.train_position_predictor \
    --time-offset ${time_offset} \
    --val-dataset data/map_extraction/positions_maps/loopnav-final-mp3d-blind_val-for-training.lmdb \
    --train-dataset data/map_extraction/positions_maps/loopnav-final-mp3d-blind_train.lmdb \
    --chance-run False \
    --mode "eval"
