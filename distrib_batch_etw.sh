#!/bin/bash
#SBATCH --job-name=navigation-analysis-habitat
#SBATCH --output=/checkpoint/%u/jobs/job.%j.out
#SBATCH --error=/checkpoint/%u/jobs/job.%j.err
#SBATCH --gres=gpu:8
#SBATCH --nodes 4
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --mem-per-cpu=5625MB
#SBATCH --partition=learnfair
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@300
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
ENV_NAME="gibson-2plus-resnet50-dpfrl-depth"
# ENV_NAME="mp3d-gibson-all-loopnav-stop-grad"
# ENV_NAME="gibson-public-50-single-GPU-depth"
# ENV_NAME="gibson-2plus-se-neXt101-lstm1024-long-depth"
# ENV_NAME="mp3d-gibson-all-loopnav-noreturn-baseline-blind"
# ENV_NAME="mp3d-gibson-50-online-long-depth"
# ENV_NAME="gibson-public-flee-pointnav-ftune-rgb-r${SLURM_ARRAY_TASK_ID}"
ENV_NAME="gibson-2plus-resnet50-frn-depth"
ENV_NAME="mp3d-gibson-all-loopnav-stage-2-trained-state-blind"
ENV_NAME="testing"
CHECKPOINT="data/checkpoints/${ENV_NAME}"

module purge
module load cuda/10.0
module load cudnn/v7.6-cuda.10.0
module load NCCL/2.4.8-1-cuda.10.0
module load openmpi/4.0.1/gcc.7.4.0-git_patch#6654

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

printenv | grep SLURM
set -x
srun --mpi=pmix_v3 \
    python -u -m nav_analysis.train_ppo_gossip \
    --extra-confs \
    nav_analysis/configs/experiments/mp3d-gibson-2plus.pointnav.yaml \
    --opts \
    "logging.log_file=${EXP_DIR}/log.txt" \
    "logging.checkpoint_folder=${CHECKPOINT}" \
    "logging.tensorboard_dir=runs/${ENV_NAME}" \
    # nav_analysis/configs/experiments/mp3d-gibson-2plus.speedmaster.pointnav.yaml \
    # nav_analysis/configs/experiments/pointnav-rgb/resnet50-rgb.yaml \
    # nav_analysis/configs/experiments/loopnav/loopnav_sparse_pg_blind.yaml \
    # nav_analysis/configs/experiments/loopnav/stage_2_trained_state.yaml \

