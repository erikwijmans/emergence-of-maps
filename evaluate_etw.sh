#!/usr/bin/env bash
#SBATCH --job-name=navigation-analysis-habitat-eval
#SBATCH --output=/checkpoint/%u/jobs/job.%j.out
#SBATCH --error=/checkpoint/%u/jobs/job.%j.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem=100GB
#SBATCH --partition=dev
#SBATCH --time=24:00:00

module purge
module load cuda/10.0
module load cudnn/v7.4-cuda.10.0
module load NCCL/2.4.2-1-cuda.10.0

export PYTHONPATH=$(pwd)/habitat-api-navigation-analysis:${PYTHONPATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

SIM_GPU_IDS="0"
PTH_GPU_ID="0"
SENSOR_TYPES="RGB_SENSOR"
NUM_PROCESSES=4
CHECKPOINT_MODEL_DIR="data/checkpoints/rgbd_with_norm"
ENV_NAME=$(basename ${CHECKPOINT_MODEL_DIR})
# ENV_NAME="testing"
MAX_EPISODE_TIMESTEPS=2000
TASK_CONFIG="tasks/gibson.pointnav.yaml"
NAV_TASK="pointnav"

if [ ${NAV_TASK} == "loopnav" ]
then
    echo "FIX ME"
else
    export LOG_FILE="/private/home/erikwijmans/projects/navigation-analysis-habitat/data/eval/eval.${ENV_NAME}.pointnav.log"
    # export LOG_FILE="/private/home/akadian/navigation-analysis/navigation-analysis-habitat/evaluate.blind.loopnav.T_S.log"
    # export POSITIONS_FILE="/private/home/akadian/navigation-analysis/navigation-analysis-habitat/sts_episodes.pkl"
fi
rm ${LOG_FILE}

COUNT_TEST_EPISODES=994
VIDEO=0
OUT_DIR_VIDEO="/private/home/akadian/navigation-analysis/navigation-analysis-habitat/eval-videos-${NAV_TASK}"

# for i in 1 2 4 8 16 32 64 96 128 192 256 512
# do
python -u -m nav_analysis.evaluate_ppo \
    --checkpoint-model-dir ${CHECKPOINT_MODEL_DIR} \
    --sim-gpu-ids ${SIM_GPU_IDS} \
    --pth-gpu-id ${PTH_GPU_ID} \
    --num-processes ${NUM_PROCESSES} \
    --log-file ${LOG_FILE} \
    --count-test-episodes ${COUNT_TEST_EPISODES} \
    --video ${VIDEO} \
    --out-dir-video ${OUT_DIR_VIDEO} \
    --nav-task ${NAV_TASK} \
    --tensorboard-dir "runs/${ENV_NAME}" \
    --nav-env-verbose 0 \
    # --max-memory-length ${i}
# done
