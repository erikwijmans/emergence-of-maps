#!/bin/bash
#SBATCH --job-name=navigation-analysis-habitat
#SBATCH --output=/checkpoint/%u/jobs/job.%A_%a.out
#SBATCH --error=/checkpoint/%u/jobs/job.%A_%a.err
#SBATCH --gpus-per-task=1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=learnlab,learnfair
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@300
#SBATCH --open-mode=append
#SBATCH --array=0-2

echo ${PYTHONPATH}

echo "Using setup for Erik"
. /private/home/erikwijmans/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate nav-analysis-base

SCENES=(RPmz2sHmrrY Scioto jtcxE69GiFV)
SCENE=${SCENES[${SLURM_ARRAY_TASK_ID}]}


BASE_EXP_DIR="/checkpoint/erikwijmans/exp-dir"
CURRENT_DATETIME="`date +%Y_%m_%d_%H_%M_%S`";
EXP_DIR="${BASE_EXP_DIR}/exp_habitat_api_navigation_analysis_datetime_${CURRENT_DATETIME}"
mkdir -p ${EXP_DIR}
ENV_NAME="mdp/${SCENE}"
CHECKPOINT="data/checkpoints/${ENV_NAME}"

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}
export GLOG_minloglevel=3
export MAGNUM_LOG=quiet
unset NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

printenv | grep SLURM
set -x
srun --kill-on-bad-exit=1 \
    python -u -m nav_analysis.train_ppo_distrib \
    --extra-confs \
    nav_analysis/configs/experiments/loopnav/loopnav_sparse_pg_blind.yaml \
    nav_analysis/configs/experiments/loopnav/stage_1.yaml \
    --opts \
    "logging.log_file=${EXP_DIR}/log.txt" \
    "logging.checkpoint_folder=${CHECKPOINT}" \
    "logging.tensorboard_dir=runs/${ENV_NAME}" \
    "task.task_config=tasks/mdp/mdp.pointnav.yaml" \
    "task.split=${SCENE}" \
    "ppo.num_processes=32"  \
    "ppo.num_updates=100000"

    # nav_analysis/configs/experiments/loopnav/loopnav_sparse_pg_blind.yaml \
    # nav_analysis/configs/experiments/loopnav/stage_1.yaml \
