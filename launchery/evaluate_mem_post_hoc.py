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

warnings.simplefilter(action="ignore", category=FutureWarning)

from nav_analysis.evaluate_ppo import eval_checkpoint, ModelEvaluator

fairtask.queue.TASK_MAX_RETRIES = 30

slurm_q = fairtask_slurm.SLURMQueueConfig(
    name="eval-mem-len",
    num_workers_per_node=2,
    cpus_per_worker=5,
    mem_gb_per_worker=50,
    num_jobs=10,
    partition="dev",
    maxtime_mins=int(6 * 60),
    gres="gpu:1",
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


model = "/private/home/erikwijmans/projects/navigation-analysis-habitat/data/checkpoints/mp3d-gibson-all-loopnav-stage-1-v5-blind/ckpt.278.pth"

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
    --nav-task pointnav
    --tensorboard-dir 'runs/memory-length/{env_name}'
    --nav-env-verbose 0
""".replace(
    "\n", " "
)


eval_checkpoint_slurm = qs.task("slurm")(eval_checkpoint)


@qs.task("gpu1")
def gpu1_q(*args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    return eval_checkpoint(*args)


@qs.task("gpu2")
def gpu2_q(*args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    return eval_checkpoint(*args)


async def main():
    parser = ModelEvaluator.build_parser()
    env_name = "none"

    tasks = []
    for mlen in tqdm.tqdm([1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048]):
        args = parser.parse_args(
            args=shlex.split(
                ARGS_STR_TEMPLATE.format(
                    checkpoint_dir=osp.dirname(model), env_name=env_name,
                )
            )
        )
        args.max_memory_length = mlen
        for _ in range(3):
            t = eval_checkpoint_slurm(args, model)

            tasks.append(t)

    data = dict(mem_len=[], spl=[], success=[])
    with tqdm.tqdm(total=len(tasks)) as pbar:
        for task in asyncio.as_completed(tasks):
            stats_means, stats_episodes, trained_args, num_frames, args = await task

            for _, v in stats_episodes.items():
                data["mem_len"].append(args.max_memory_length)
                data["spl"].append(v["spl"])
                data["success"].append(v["success"])

            pbar.update()

    with open("spl_vs_mem_len_post_hoc.msg", "wb") as f:
        msgpack_numpy.pack(data, f, use_bin_type=True)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
