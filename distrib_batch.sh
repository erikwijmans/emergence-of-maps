#!/bin/bash
#SBATCH --job-name=navigation-analysis-habitat
#SBATCH --output=/checkpoint/akadian/jobs/job.%j.out
#SBATCH --error=/checkpoint/akadian/jobs/job.%j.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
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
    BASE_EXP_DIR="/checkpoint/akadian/exp-dir"
    CHECKPOINT="${BASE_EXP_DIR}/checkpoints"
    mkdir -p ${CHECKPOINT}
elif [${USER} == "erikwijmans"]
then
    echo "Using setup for Erik"
    . /private/home/erikwijmans/miniconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate hsim
    BASE_EXP_DIR="/checkpoint/erikwijmans/exp-dir"
    CHECKPOINT="data/checkpoints/gibson_depth_1k"
fi

ENV_NAME="pointnav_gibson_depth"
SENSORS="DEPTH_SENSOR"
PG_SENSOR_TYPE="DENSE"
BLIND=0
RNN_TYPE="LSTM"
CURRENT_DATETIME="`date +%Y_%m_%d_%H_%M_%S`";

module purge
module load cuda/10.0
module load cudnn/v7.4-cuda.10.0
module load NCCL/2.4.2-1-cuda.10.0

export PYTHONPATH=$(pwd):$(pwd)/habitat-api-navigation-analysis:${PYTHONPATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

EXP_DIR="${BASE_EXP_DIR}/exp.habitat_api_navigation_analysis.datetime_${CURRENT_DATETIME}"
mkdir -p ${EXP_DIR}

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
    --num-steps 128 \
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
    --env-name "${ENV_NAME}"
