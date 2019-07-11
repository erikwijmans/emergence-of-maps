#!/bin/bash
#SBATCH --job-name=benchmarking
#SBATCH --output=benchmarking/{NGPU}.out
#SBATCH --error=benchmarking/{NGPU}.err
#SBATCH --gres gpu:{NTASKS}
#SBATCH --nodes {NNODES}
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node {NTASKS}
#SBATCH --mem-per-cpu=5625MB
#SBATCH --partition=learnfair
#SBATCH --comment="benchmarking"
#SBATCH --time=2:00:00
#SBATCH --signal=USR1@300
#SBATCH --constraint=volta32gb
#SBATCH --exclusive

echo "Using setup for Erik"
. /private/home/erikwijmans/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate hsim


module purge
module load cuda/10.0
module load cudnn/v7.4-cuda.10.0
module load NCCL/2.4.7-1-cuda.10.0

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${{LD_LIBRARY_PATH}}
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
set -x
srun python -u -m nav_analysis.benchmark_distrib_main {TASK} {FKEY}

