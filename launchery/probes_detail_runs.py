import warnings
import shlex
import os.path as osp
import tqdm
import msgpack_numpy
import glob

warnings.simplefilter(action="ignore", category=FutureWarning)

from nav_analysis.evaluate_ppo import eval_checkpoint, ModelEvaluator


ARGS_STR_TEMPLATE = """
    --exit-immediately
    --checkpoint-model-dir {checkpoint_dir}
    --sim-gpu-ids 0
    --pth-gpu-id 0
    --num-processes 18
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


def main():
    parser = ModelEvaluator.build_parser()
    env_name = "none"
    run_name_template = "mp3d-gibson-all-{task}-stage-2-trained-state-final-run_*-blind"

    task_types = ["loopnav", "teleportnav"]

    best_detailed_stats_by_name = {}
    best_by_name = {}

    for task in tqdm.tqdm(task_types):
        for i, checkpoint_dir in enumerate(
            glob.glob(
                osp.join("data/checkpoints/probes", run_name_template.format(task=task))
            )
        ):
            args = parser.parse_args(
                args=shlex.split(
                    ARGS_STR_TEMPLATE.format(
                        checkpoint_dir=checkpoint_dir, env_name=env_name, task=task,
                    )
                )
            )
            for model in tqdm.tqdm(glob.glob(osp.join(checkpoint_dir, "*"))):
                name = str(task, i)
                (stats_means, stats_episodes, _, _, _) = eval_checkpoint(args, model)


                if (
                    name not in best_by_name
                    or stats_means["stage_2_spl"].mean > best_by_name[name]
                ):
                    best_by_name[name] = stats_means["stage_2_spl"].mean
                    best_detailed_stats_by_name[name] = stats_episodes

    with open("probe_detailed_stats.msg", "wb") as f:
        msgpack_numpy.pack(best_detailed_stats_by_name, f, use_bin_type=True)


if __name__ == "__main__":
    main()
