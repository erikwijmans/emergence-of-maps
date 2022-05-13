#!/bin/bash
#SBATCH --job-name=navigation-analysis-habitat
#SBATCH --output=/checkpoint/%u/jobs/job.%A_%a.out
#SBATCH --error=/checkpoint/%u/jobs/job.%A_%a.err
#SBATCH --gpus-per-task=1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=learnlab,learnfair
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@300
#SBATCH --open-mode=append
#SBATCH --array=0-4

echo ${PYTHONPATH}

echo "Using setup for Erik"
. /private/home/erikwijmans/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate nav-analysis-base

SCENES=(RPmz2sHmrrY Scioto jtcxE69GiFV)
SCENES=(Scioto jtcxE69GiFV)

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}
export GLOG_minloglevel=3
export MAGNUM_LOG=quiet
unset NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME

SIM_GPU_IDS="0"
PTH_GPU_ID="0"

BASE_EXP_DIR="/checkpoint/erikwijmans/exp-dir"
CURRENT_DATETIME="`date +%Y_%m_%d_%H_%M_%S`";
EXP_DIR="${BASE_EXP_DIR}/exp_habitat_api_navigation_analysis_datetime_${CURRENT_DATETIME}"
mkdir -p ${EXP_DIR}

for SCENE in "${SCENES[@]}"
do

    ENV_NAME="mdp/${SCENE}"
    CHECKPOINT="data/checkpoints/${ENV_NAME}"

    TASK_CONFIG="tasks/mdp/pomdp.pointnav.yaml"
    OUT_DIR_VIDEO="${CHECKPOINT}/eval/videos-pomdp"

    python -u -m nav_analysis.evaluate_ppo \
        --checkpoint-model-dir data/checkpoints/demo_probe.pth \
        --sim-gpu-ids ${SIM_GPU_IDS} \
        --pth-gpu-id ${PTH_GPU_ID} \
        --num-processes 1 \
        --log-file "${CHECKPOINT}/eval.log" \
        --count-test-episodes 100 \
        --video 1 \
        --out-dir-video ${OUT_DIR_VIDEO} \
        --eval-task-config ${TASK_CONFIG} \
        --nav-task "teleportnav" \
        --tensorboard-dir "runs/${ENV_NAME}" \
        --nav-env-verbose 0 \
        --split "${SCENE}" \
        --exit-immediately

    OUT_DIR_VIDEO="${CHECKPOINT}/eval/videos-mdp"
    TASK_CONFIG="tasks/mdp/mdp.pointnav.yaml"

    python -u -m nav_analysis.evaluate_ppo \
        --checkpoint-model-dir ${CHECKPOINT}/ckpt.165.pth \
        --sim-gpu-ids ${SIM_GPU_IDS} \
        --pth-gpu-id ${PTH_GPU_ID} \
        --num-processes 1 \
        --log-file "${CHECKPOINT}/eval.log" \
        --count-test-episodes 100 \
        --video 1 \
        --out-dir-video ${OUT_DIR_VIDEO} \
        --eval-task-config ${TASK_CONFIG} \
        --nav-task "pointnav" \
        --tensorboard-dir "runs/${ENV_NAME}" \
        --nav-env-verbose 0 \
        --split "${SCENE}" \
        --exit-immediately

done
