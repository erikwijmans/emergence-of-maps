#!/bin/bash
#SBATCH --job-name=navigation-analysis-habitat
#SBATCH --output=/checkpoint/%u/jobs/job.%j.out
#SBATCH --error=/checkpoint/%u/jobs/job.%j.err
#SBATCH --gres gpu:8
#SBATCH --nodes 4
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --mem-per-cpu=5625MB
#SBATCH --partition=learnfair
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append
echo "#SBATCH --constraint=volta32gb"

echo ${PYTHONPATH}

echo "Using setup for Erik"
. /private/home/erikwijmans/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate nav-analysis-base


BASE_EXP_DIR="/checkpoint/erikwijmans/exp-dir"
CURRENT_DATETIME="`date +%Y_%m_%d_%H_%M_%S`";
EXP_DIR="${BASE_EXP_DIR}/exp_habitat_api_navigation_analysis_datetime_${CURRENT_DATETIME}"
mkdir -p ${EXP_DIR}
ENV_NAME="gibson-challenge-mp3d-gibson-se-neXt25-depth"
ENV_NAME="gibson-all-se-neXt25-depth"
ENV_NAME="mp3d-gibson-2plus-se-resneXt50-long-depth"
# ENV_NAME="gibson-public-50-single-GPU-depth"
# ENV_NAME="gibson-2plus-se-neXt101-lstm1024-long-depth"
ENV_NAME="mp3d-gibson-all-loopnav-noreturn-baseline-blind"
# ENV_NAME="mp3d-gibson-50-online-long-depth"
CHECKPOINT="data/checkpoints/${ENV_NAME}"

module purge
module load cuda/10.0
module load cudnn/v7.4-cuda.10.0
module load NCCL/2.4.7-1-cuda.10.0

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
set -x
srun python -u -m nav_analysis.train_ppo_distrib \
    --extra-confs configs/experiments/loopnav_no_return_baseline.yaml \
    --opts \
    "logging.log_file=${EXP_DIR}/log.txt" \
    "logging.checkpoint_folder=${CHECKPOINT}" \
    "logging.tensorboard_dir=runs/${ENV_NAME}" \
    "model.two_headed=False"
