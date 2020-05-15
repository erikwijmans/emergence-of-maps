#!/bin/bash
#SBATCH --job-name=navigation-analysis-habitat
#SBATCH --output=/checkpoint/%u/jobs/job.%A_%a.out
#SBATCH --error=/checkpoint/%u/jobs/job.%A_%a.err
#SBATCH --gres=gpu:8
#SBATCH --nodes 2
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=learnfair,scavenge
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@300
#SBATCH --open-mode=append
#SBATCH --constraint=volta32gb
#SBATCH --array=0-59

echo ${PYTHONPATH}

echo "Using setup for Erik"
. /private/home/erikwijmans/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate nav-analysis-base

# DEAD_RUNS_ARRAY=(20 21 22 30 31 42 43 55 56)

PROBE_TASKS=(loopnav teleportnav)
STATE_TYPES=(trained random zero)
INPUT_TYPES=(blind "no-inputs")

run_id=${SLURM_ARRAY_TASK_ID}
# run_id=${DEAD_RUNS_ARRAY[${SLURM_ARRAY_TASK_ID}]}

repeat_idx=$((${run_id} / 60))
run_id=$((${run_id} % 60))

base_agent_idx=$((${run_id} / 12 ))
run_id=$((${run_id} % 12 ))

input_idx=$(( ${run_id} / 6 ))
run_id=$((${run_id} % 6))

task_idx=$((${run_id} / 3))
state_idx=$((${run_id} % 3))

state_type=${STATE_TYPES[${state_idx}]}
task_type=${PROBE_TASKS[${task_idx}]}
input_type=${INPUT_TYPES[${input_idx}]}



BASE_EXP_DIR="/checkpoint/erikwijmans/exp-dir"
CURRENT_DATETIME="`date +%Y_%m_%d_%H_%M_%S`";
EXP_DIR="${BASE_EXP_DIR}/exp_habitat_api_navigation_analysis_datetime_${CURRENT_DATETIME}"
mkdir -p ${EXP_DIR}
ENV_NAME="mp3d-gibson-all-${task_type}-stage-2-${state_type}-state-final-run_${repeat_idx}_${base_agent_idx}-${input_type}"
CHECKPOINT="data/checkpoints/probes/${ENV_NAME}"

module purge
module load cuda/10.1
module load cudnn/v7.6.5.32-cuda.10.1
module load NCCL/2.5.6-1-cuda.10.1
module load openmpi/4.0.1/gcc.7.4.0-git_patch#6654

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}
export GLOG_minloglevel=3
export MAGNUM_LOG=quiet


configs=(nav_analysis/configs/experiments/loopnav/loopnav_sparse_pg_blind.yaml \
    nav_analysis/configs/experiments/loopnav/stage_2_${state_type}_state.yaml)

if [ ${task_type} = "teleportnav" ]; then
    configs+=(nav_analysis/configs/experiments/teleportnav/teleport_stage_2.yaml)
fi

extra_opts=()
if [ ${input_type} = "no-inputs" ]; then
    extra_opts+=(task.loopnav_give_return_inputs=False \
        ppo.num_updates=75000)
fi

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

printenv | grep SLURM
set -x
srun --mpi=pmix_v3 --kill-on-bad-exit=1 \
    python -u -m nav_analysis.train_ppo_distrib \
    --extra-confs \
    ${configs[@]} \
    --opts \
    "logging.log_file=${EXP_DIR}/log.txt" \
    "logging.checkpoint_folder=${CHECKPOINT}" \
    "logging.tensorboard_dir=runs/probes/${ENV_NAME}" \
    stage_2_args.stage_1_model=/private/home/erikwijmans/projects/navigation-analysis-habitat/data/checkpoints/mp3d-gibson-all-loopnav-stage-1-final-run_${base_agent_idx}-blind/ckpt.175.pth \
    ${extra_opts[@]}

