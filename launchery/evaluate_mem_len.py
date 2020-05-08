import submitit
from nav_analysis.evaluate_ppo import ModelEvaluator
import glob
import shlex
import os.path as osp
import tqdm

ENV_NAME_TEMPLATE = "memory-length/mp3d-gibson-all-pointnav-mem_*-blind"

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


def build_all_args():
    parser = ModelEvaluator.build_parser()

    for ckpt_dir in glob.glob(osp.join("data/checkpoints", ENV_NAME_TEMPLATE)):
        mem_len = int(ckpt_dir.split("mem_")[1].split("-")[0])
        #  if mem_len != 256:
        #  continue

        yield parser.parse_args(
            args=shlex.split(
                ARGS_STR_TEMPLATE.format(
                    checkpoint_dir=ckpt_dir, env_name=osp.basename(ckpt_dir),
                )
            )
        )


if __name__ == "__main__":
    executor = submitit.AutoExecutor(
        folder="/checkpoint/erikwijmans/submitit/%j", cluster="slurm"
    )
    executor.update_parameters(
        mem_gb=50,
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=10,
        nodes=1,
        timeout_min=60 * 24,
        slurm_partition="learnfair,scavenge",
        name="nav-analysis-mem-len-eval",
        slurm_signal_delay_s=60,
    )

    jobs = []
    with executor.batch():
        for args in build_all_args():
            jobs.append(executor.submit(ModelEvaluator(), args))

    print("Started", len(jobs), "jobs")

    for job in tqdm.tqdm(jobs):
        print(job.results())
