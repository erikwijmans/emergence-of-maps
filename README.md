Emergence of Maps in the Memories of Blind Navigation Agents

## Setup

### Software

Tested on Ubuntu 16, 18, and 20.

Install miniconda/anaconda: https://docs.conda.io/en/latest/miniconda.html


Install environment: 

```
conda env create -f environment.yml
```


Add habitat API to PYTHONPATH

```
export PYTHONPATH=$(pwd)/habitat-api-emergence-of-maps:${PYTHONPATH}
```


Expected installation time: 10 minutes


### Data folder

All data will be put into a `data/` folder in the root directory of the repo.  One will be created automatically if it is missing, but if your system setup requires that data is kept on a different partition/mount, create a folder there and symlink it to `data/` here.

## Demo


### Agent and Probe

To demo training an agent and a probe, first download the Habitat demo data:

```
wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip
unzip habitat-test-scenes.zip

python scripts/unbundle_demo_data.py
```

Then train an agent with

```
python -u -m nav_analysis.train_ppo_distrib \
    --extra-confs \
    nav_analysis/configs/experiments/loopnav/loopnav_sparse_pg_blind.yaml \
    nav_analysis/configs/experiments/loopnav/stage_1.yaml \
    nav_analysis/configs/experiments/demo.yaml
```

This should take around an hour on a 1 GPU machine and get to about 95% success.

Then generate probe training episodes with
```
python -u -m nav_analysis.rl.build_stage_2_episodes \
    --model-paths-glob data/checkpoints/demo/ckpt.0.pth \
    --sample-per-scene 1000.0
```
This should take around 5 minutes

Then train a loopnav probe with
```
python -u -m nav_analysis.train_ppo_distrib \
    --extra-confs \
    nav_analysis/configs/experiments/loopnav/loopnav_sparse_pg_blind.yaml \
    nav_analysis/configs/experiments/loopnav/stage_2_trained_state.yaml \
    nav_analysis/configs/experiments/demo.yaml \
    --opts \
    "logging.checkpoint_folder=data/checkpoints/demo-loopnav" \
    "logging.tensorboard_dir=runs/demo-loopnav"
```
This should take around an hour and reach >98% success.

and finally evaluate the agent/probe pair with
```
python -u -m nav_analysis.evaluate_ppo \
    --exit-immediately \
    --checkpoint-model-dir data/checkpoints/demo-loopnav \
    --sim-gpu-ids 0 \
    --pth-gpu-id 0 \
    --num-processes 2 \
    --log-file demo.eval.log \
    --count-test-episodes 100 \
    --video 0 \
    --out-dir-video demo_video \
    --eval-task-config tasks/loopnav/demo.loopnav.yaml \
    --nav-task loopnav \
    --tensorboard-dir "runs/demo-loopnav" \
    --nav-env-verbose 0
```

Evaluation should take around 2 minutes.  Stage 1 SPL is the agent's SPL and Stage 2 SPL is the probe's SPL.  Probe SPL will be higher than agent SPL.

### Map Decoding

Next, create the episodes to train the position predictors and map decoder.


```
python -u -m nav_analysis.map_extraction.data.build_positions_dataset \
    --model-path data/checkpoints/demo-loopnav/ckpt.0.pth \
    --num-processes 2 \
    --task-config tasks/loopnav/demo.loopnav.yaml \
    --num-val 50 \
    --num-train 200 \
    --output-path data/map_extraction/demo
```

This should take around 10 minutes


Then create the map cache

```
python -u -m nav_analysis.map_extraction.data.build_visited_dataset \
    --positions-output data/map_extraction/demo
```

This should take under a minute


Then train the map predictor
```
python -u -m nav_analysis.map_extraction.training.train_visited_predictor \
    --dset-path data/map_extraction/demo \
    --model-save-name data/demo-occ-predictor \
    --epochs 10
```

This should take a few minutes and reach around 40 Val IoU

### Visitation Position Predictor


A predictor for where the agent will be in `k` steps (5 steps in the past with this demo) can be trained as follows:

```
python -u -m nav_analysis.map_extraction.training.train_position_predictor \
    --time-offset -5 \
    --val-dataset data/map_extraction/demo_val.lmdb \
    --train-dataset data/map_extraction/demo_train.lmdb \
    --chance-run False \
    --mode "train"
```

This should take a few minutes and reach a best val L2 error of less than 0.2


We will provide the full data at publication.

