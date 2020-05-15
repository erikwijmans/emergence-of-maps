import warnings
import shlex
import os.path as osp
import msgpack_numpy

warnings.simplefilter(action="ignore", category=FutureWarning)

from nav_analysis.evaluate_ppo import eval_checkpoint, ModelEvaluator


models = dict(loopnav=[], teleportnav=[])


models["teleportnav"] = [
    "data/checkpoints/probes/mp3d-gibson-all-teleportnav-stage-2-trained-state-final-run_0_{}-no-inputs/ckpt.{}.pth".format(
        *v
    )
    for v in [(0, 145), (1, 121), (2, 147), (0, 148), (4, 137)]
]
models["loopnav"] = [
    "data/checkpoints/probes/mp3d-gibson-all-loopnav-stage-2-trained-state-final-run_0_{}-no-inputs/ckpt.{}.pth".format(
        *v
    )
    for v in [(0, 148), (2, 145), (3, 140), (4, 110)]
]


ARGS_STR_TEMPLATE = """
    --exit-immediately
    --checkpoint-model-dir {checkpoint_dir}
    --sim-gpu-ids 0
    --pth-gpu-id 0
    --num-processes 14
    --log-file
    '/private/home/erikwijmans/projects/navigation-analysis-habitat/data/eval/eval.{env_name}.log'
    --count-test-episodes 1008
    --video 0
    --out-dir-video
    '/private/home/erikwijmans/projects/navigation-analysis-habitat/data/eval/videos/{env_name}'
    --eval-task-config
    'tasks/loopnav/mp3d-gibson.loopnav.yaml'
    --nav-task {task}
    --tensorboard-dir 'runs/{env_name}'
    --nav-env-verbose 0
""".replace(
    "\n", " "
)

if __name__ == "__main__":
    parser = ModelEvaluator.build_parser()
    env_name = "none"

    detailed_stats = dict()
    for task, models in models.items():

        args = parser.parse_args(
            args=shlex.split(
                ARGS_STR_TEMPLATE.format(
                    checkpoint_dir=osp.dirname(models[0]), env_name=env_name, task=task
                )
            )
        )

        task_stats = []
        for model in models:
            _, stats_episodes, *_ = eval_checkpoint(args, model)

            for _, v in stats_episodes.items():
                task_stats.append(v)

        detailed_stats[task] = task_stats

    with open("no_input_detailed_stats.msg", "wb") as f:
        msgpack_numpy.pack(detailed_stats, f, use_bin_type=True)
