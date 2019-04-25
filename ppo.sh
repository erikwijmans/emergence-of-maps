#!/usr/bin/env bash

export PYTHONPATH=$(pwd):$(pwd)/habitat-api-navigation-analysis:${PYTHONPATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

python -u src/train_ppo.py \
    --use-gae \
    --sim-gpu-id 0 \
    --pth-gpu-id 1 \
    --lr 2.5e-4 \
    --clip-param 0.1 \
    --value-loss-coef 0.5 \
    --num-processes 2 \
    --num-steps 128 \
    --num-mini-batch 1 \
    --num-updates 10000 \
    --use-linear-lr-decay \
    --use-linear-clip-decay \
    --entropy-coef 0.01 \
    --log-file "/checkpoint/akadian/debug-exps/d0/debug.0.log" \
    --log-interval 1 \
    --checkpoint-folder "/checkpoint/akadian/debug-exps/d0/checkpoints" \
    --checkpoint-interval 5 \
    --sensors "RGB_SENSOR,DEPTH_SENSOR" \
    --shuffle-interval 500 \
    --task-config "tasks/gibson.pointnav.yaml" \
    --nav-task "loopnav" \
    --max-episode-timesteps 1000 \
    --pointgoal-sensor-dimensions 3 \
    --pointgoal-sensor-format "CARTESIAN" \

--use-gae --sim-gpu-id 0 --pth-gpu-id 1 --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 2 --num-steps 128 --num-mini-batch 1 --num-updates 10000 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --log-file "/checkpoint/akadian/debug-exps/d0/debug.0.log" --log-interval 1 --checkpoint-folder "/checkpoint/akadian/debug-exps/d0/checkpoints" --checkpoint-interval 5 --sensors "RGB_SENSOR,DEPTH_SENSOR" --shuffle-interval 500 --task-config "tasks/gibson.pointnav.yaml" --nav-task "loopnav" --max-episode-timesteps 1000 --pointgoal-sensor-dimensions 3 --pointgoal-sensor-format "CARTESIAN"