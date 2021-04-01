import warnings
import shlex
import os.path as osp
import tqdm
import fairtask
import fairtask_slurm
import atexit
import asyncio
import msgpack_numpy
import os
import glob

warnings.simplefilter(action="ignore", category=FutureWarning)

from nav_analysis.evaluate_ppo import eval_checkpoint, ModelEvaluator

fairtask.queue.TASK_MAX_RETRIES = 30

slurm_q = fairtask_slurm.SLURMQueueConfig(
    name="probe-detail-eval",
    num_workers_per_node=2,
    cpus_per_worker=5,
    mem_gb_per_worker=25,
    gres="gpu:1",
    num_jobs=40,
    partition="learnfair",
    maxtime_mins=int(24 * 60),
    log_directory="/checkpoint/{user}/fairtask",
    output="slurm-%j.out",
    error="slurm-%j.err",
)

qs = fairtask.TaskQueues(
    {
        "gpu1": fairtask.LocalQueueConfig(num_workers=5),
        "gpu2": fairtask.LocalQueueConfig(num_workers=5),
        "slurm": slurm_q,
    },
    no_workers=False,
)


@atexit.register
def close_queues():
    qs.close()


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


@qs.task("slurm")
def eval_checkpoint_slurm(name, args):
    return name, eval_checkpoint(*args)


@qs.task("gpu1")
def eval_checkpoint_gpu1(name, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    return name, eval_checkpoint(*args)


@qs.task("gpu2")
def eval_checkpoint_gpu1(name, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    return name, eval_checkpoint(*args)


async def main():
    parser = ModelEvaluator.build_parser()
    env_name = "none"
    run_name_template = "mp3d-gibson-all-{task}-stage-2-trained-state-final-run_*-blind"

    task_types = ["loopnav", "teleportnav"]

    best_detailed_stats_by_name = {}
    best_by_name = {}

    tasks = []
    for task in task_types:
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
            for model in glob.glob(osp.join(checkpoint_dir, "*")):
                tasks.append(eval_checkpoint_slurm(str(task, i), (args, model)))

    with tqdm.tqdm(total=len(tasks)) as pbar:
        for task in asyncio.as_completed(tasks):
            name, (stats_means, stats_episodes, _, _, _) = await task

            if (
                name not in best_by_name
                or stats_means["stage_2_spl"].mean > best_by_name[name]
            ):
                best_by_name[name] = stats_means["stage_2_spl"].mean
                best_detailed_stats_by_name[name] = stats_episodes

            pbar.update()

    with open("probe_detailed_stats.msg", "wb") as f:
        msgpack_numpy.pack(best_detailed_stats_by_name, f, use_bin_type=True)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
