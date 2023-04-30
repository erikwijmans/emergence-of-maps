# Emergence of Maps in the Memories of Blind Navigation Agents

This repository contains the code used in our ICLR 2023 paper, Emergence of Maps in the Memories of Blind Navigation Agents.
The intent of this repository is for documentations purposes. If you'd like to build upon our experiments, we highly suggest
you use this repository as a reference but build upon the latest version of habitat-lab and habitat-sim. Those have both advanced and improved
considerably since we began this work.


## Key files

* `nav_analysis/train_ppo_distrib.py` This contains the DD-PPO implementation used for training agents and probes.

* `nav_analysis/rl/build_stage_2_episodes.py`. This contains the code to build the episodes for training probes. (Referred to as stage 2 episodes.)

* `nav_analysis/evaluate_ppo.py`. This contains the code to evaluate agents and probes.

* `nav_analysis/map_extraction/data/build_positions_dataset.py`. This contains the code to build the dataset needed to prediction agent positions and decoding maps.

* `nav_analysis/map_extraction/data/build_visited_dataset.py`. This creates a cache for training the top-down visitation models and occupancy prediction models.

* `nav_analysis/map_extraction/training/train_visited_predictor.py`. This trains the top-down visited and occupancy map models. The top-down visited models are for easier visualization. See `nav_analysis/map_extraction/viz/past_prediction_figure.py` for visualization of past prediction and `nav_analysis/map_extraction/viz/top_down_occ_figure.py` for a visualization of occupancy prediction.

* `nav_analysis/map_extraction/training/train_position_predictor.py`. This trains the position predictors used by `nav_analysis/map_extraction/training/calibrated_position_predictor.py` and `visualizations/calibrated_position_predictor_viz.py`.

* `nav_analysis/map_extraction/data_annotation` contains the web-app we used to manually annotation excursions.

* `nav_analysis/map_extraction/data/build_collision_detection_dataset.py` builds a collision detection dataset. This can then be used with `nav_analysis/map_extraction/training/train_collision_detector.py` to train collision detectors.

In the `launchery` folder, there are helper scripts for launching experiments. Most of these would highly benefit from parallelization depending on your cluster environment. Some are already parallelized using SLURM.

## Key Terms

* LoopNav. This corresponds to the T->S probe.

* TeleportNav. This corresponds to the S->T probe.

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

Then train a T->S (loopnav) probe with
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

