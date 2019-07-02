import collections
import json
import random
import shlex
import subprocess

import tqdm

template = r"""
echo "Using setup for Erik"
. /private/home/erikwijmans/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate nav-analysis-base


SENSORS="DEPTH_SENSOR"
PG_SENSOR_TYPE="DENSE"
PG_SENSOR_DIMENSIONS=2
PG_FORMAT="POLAR"
RNN_TYPE="LSTM"
NAV_TASK="pointnav"
MAX_EPISODE_TIMESTEPS=500
CHECKPOINT=tmp

BLIND=0
NUM_STEPS=128

module purge
module load cuda/10.0
module load cudnn/v7.4-cuda.10.0
module load NCCL/2.4.2-1-cuda.10.0

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${{LD_LIBRARY_PATH}}
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
set -x
srun --ntasks={NTASKS} --nodes={NNODES} python -u -m nav_analysis.benchmark_distrib_work \
    --shuffle-interval 50000 \
    --use-gae \
    --sim-gpu-id 0 \
    --pth-gpu-id 0 \
    --lr 2.5e-4 \
    --clip-param 0.1 \
    --value-loss-coef 0.5 \
    --num-processes 4 \
    --num-steps ${{NUM_STEPS}} \
    --num-mini-batch 2 \
    --ppo-epoch 2 \
    --num-updates 50 \
    --entropy-coef 0.01 \
    --log-file "${{EXP_DIR}}/train.log" \
    --log-interval 25 \
    --checkpoint-folder ${{CHECKPOINT}} \
    --checkpoint-interval 200 \
    --task-config {TASK} \
    --sensors ${{SENSORS}} \
    --num-recurrent-layers 2 \
    --hidden-size 512 \
    --rnn-type "${{RNN_TYPE}}" \
    --reward-window-size 200 \
    --blind "${{BLIND}}" \
    --pointgoal-sensor-type "${{PG_SENSOR_TYPE}}" \
    --pointgoal-sensor-dimensions ${{PG_SENSOR_DIMENSIONS}} \
    --pointgoal-sensor-format ${{PG_FORMAT}} \
    --nav-task "${{NAV_TASK}}" \
    --max-episode-timesteps ${{MAX_EPISODE_TIMESTEPS}} \
    --resnet-baseplanes 32 \
    --weight-decay 0.0 \
    --backbone resnet50 \
    --tensorboard-dir "runs" \
    --backprop {BACKPROP} \
    --seed {SEED}
"""


def call(cmd):
    cmd = "bash -c '{}'".format(cmd)
    output = subprocess.check_output(
        shlex.split(cmd), stderr=open("/dev/null", "w")
    )
    return output.decode("utf-8")


def main():
    n_repeats = 10
    MP3D = "tasks/mp3d-gibson.pointnav.yaml"
    GIBSON = "tasks/gibson-public.pointnav.yaml"
    gpus = [16, 32, 64, 96, 128]
    perf_results = collections.defaultdict(list)

    for ngpu in gpus:
        params = dict(
            BACKPROP=0,
            SEED=0,
            TASK=MP3D,
            NTASKS=ngpu,
            NNODES=max(ngpu // 8, 1),
        )
        for _ in tqdm.trange(n_repeats):
            params["SEED"] = random.randint(0, int(1e5))
            res = call(template.format(**params))
            res = [l.strip() for l in res.split("\n") if len(l.strip()) > 0]
            res = json.loads(res[-1])
            perf_results[ngpu].append(res)
            tqdm.tqdm.write(str(res) + "\n")

    with open("data/perf_data/gibson-mp3d_multi_nobackprop.json", "w") as f:
        json.dump(perf_results, f)


if __name__ == "__main__":
    main()
