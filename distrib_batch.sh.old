#!/bin/bash
#SBATCH --job-name=navigation-analysis-habitat
#SBATCH --output=/checkpoint/%u/jobs/job.%j.out
#SBATCH --error=/checkpoint/%u/jobs/job.%j.err
#SBATCH --gres gpu:8
#SBATCH --nodes 8
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --mem=400GB
#SBATCH --partition=learnfair
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append

# conda setup
if [ ${USER} == "akadian" ]
then
    echo "Using setup for Abhishek"
    source activate navigation-analysis
    CURRENT_DATETIME="`date +%Y_%m_%d_%H_%M_%S`";
    EXP_DIR="/checkpoint/akadian/exp-dir/exp_habitat_api_navigation_analysis_datetime_${CURRENT_DATETIME}"
    CHECKPOINT="${EXP_DIR}/checkpoints"
    mkdir -p ${EXP_DIR}
    mkdir -p ${CHECKPOINT}
elif [ ${USER} == "erikwijmans" ]
then
    echo "Using setup for Erik"
    . /private/home/erikwijmans/miniconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate hsim
    BASE_EXP_DIR="/checkpoint/erikwijmans/exp-dir"
    EXP_DIR="${BASE_EXP_DIR}/exp_habitat_api_navigation_analysis_datetime_${CURRENT_DATETIME}"
    mkdir -p ${EXP_DIR}
    CHECKPOINT="data/checkpoints/gibson-challenge-mp3d-gibson"
fi

ENV_NAME="gibson-challenge-mp3d-gibson"
SENSORS="DEPTH_SENSOR"
PG_SENSOR_TYPE="DENSE"
PG_SENSOR_DIMENSIONS=3
PG_FORMAT="POLAR"
RNN_TYPE="LSTM"
NAV_TASK="pointnav"
MAX_EPISODE_TIMESTEPS=2000

BLIND=0
NUM_STEPS=128

module purge
module load cuda/10.0
module load cudnn/v7.4-cuda.10.0
module load NCCL/2.4.2-1-cuda.10.0

export PYTHONPATH=$(pwd):$(pwd)/habitat-api-navigation-analysis:${PYTHONPATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
set -x
srun python -u src/train_ppo_distrib.py \
    --shuffle-interval 5000 \
    --use-gae \
    --sim-gpu-id 0 \
    --pth-gpu-id 0 \
    --lr 2.5e-4 \
    --clip-param 0.1 \
    --value-loss-coef 0.5 \
    --num-processes 4 \
    --num-steps ${NUM_STEPS} \
    --num-mini-batch 2 \
    --ppo-epoch 2  \
    --num-updates 1000000 \
    --use-linear-lr-decay \
    --use-linear-clip-decay \
    --entropy-coef 0.01 \
    --log-file "${EXP_DIR}/train.log" \
    --log-interval 50 \
    --checkpoint-folder ${CHECKPOINT} \
    --checkpoint-interval 500 \
    --task-config "tasks/gibson.pointnav.yaml" \
    --sensors ${SENSORS} \
    --num-recurrent-layers 2 \
    --hidden-size 512 \
    --rnn-type "${RNN_TYPE}" \
    --reward-window-size 200 \
    --blind "${BLIND}" \
    --pointgoal-sensor-type "${PG_SENSOR_TYPE}" \
    --pointgoal-sensor-dimensions ${PG_SENSOR_DIMENSIONS} \
    --pointgoal-sensor-format ${PG_FORMAT} \
    --env-name "${ENV_NAME}" \
    --nav-task "${NAV_TASK}" \
    --max-episode-timesteps ${MAX_EPISODE_TIMESTEPS} \
    --resnet-baseplanes 32
