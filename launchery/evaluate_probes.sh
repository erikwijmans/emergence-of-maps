#!/usr/bin/env bash
#SBATCH --job-name=navigation-analysis-habitat-eval
#SBATCH --output=/checkpoint/%u/jobs/job.%j.out
#SBATCH --error=/checkpoint/%u/jobs/job.%j.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --partition=dev
#SBATCH --time=72:00:00

function kill_children()
{
    kill $(jobs -p)
    wait $(jobs -p)
}

trap kill_children SIGINT
trap kill_children SIGTERM

. /private/home/erikwijmans/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate nav-analysis-base

export PYTHONPATH=$(pwd)/habitat-api-navigation-analysis

module purge
module load cuda/10.1
module load cudnn/v7.6.5.32-cuda.10.1
module load NCCL/2.5.6-1-cuda.10.1

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

SIM_GPU_IDS="0"
PTH_GPU_ID="0"
SENSOR_TYPES="RGB_SENSOR"
NUM_PROCESSES=18
TASK_CONFIG="tasks/loopnav/mp3d.loopnav.yaml"
COUNT_TEST_EPISODES=1008
VIDEO=0

for nav_task in "loopnav" "teleportnav"; do
    for state_type in "trained" "zero" "random"; do
        CHECKPOINT_MODEL_DIR="data/checkpoints/mp3d-gibson-all-${nav_task}-stage-2-${state_type}-state-no-inputs"
        ENV_NAME=$(basename ${CHECKPOINT_MODEL_DIR})

        NAV_TASK=${nav_task}

        export LOG_FILE="/private/home/erikwijmans/projects/navigation-analysis-habitat/data/eval/eval.${ENV_NAME}.pointnav.log"
        rm ${LOG_FILE}

        OUT_DIR_VIDEO="/private/home/erikwijmans/projects/navigation-analysis-habitat/data/eval/videos/${ENV_NAME}"

        python -u -m nav_analysis.evaluate_ppo \
            --checkpoint-model-dir ${CHECKPOINT_MODEL_DIR} \
            --sim-gpu-ids ${SIM_GPU_IDS} \
            --pth-gpu-id ${PTH_GPU_ID} \
            --num-processes ${NUM_PROCESSES} \
            --log-file ${LOG_FILE} \
            --count-test-episodes ${COUNT_TEST_EPISODES} \
            --video ${VIDEO} \
            --out-dir-video ${OUT_DIR_VIDEO} \
            --eval-task-config ${TASK_CONFIG} \
            --nav-task ${NAV_TASK} \
            --tensorboard-dir "runs/${ENV_NAME}" \
            --nav-env-verbose 0 \
            &
    done
done



wait $(jobs -p)

