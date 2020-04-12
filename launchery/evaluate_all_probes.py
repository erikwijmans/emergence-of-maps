import submitit
from nav_analysis.evaluate_ppo import ModelEvaluator
import glob
import itertools
import shlex
import os.path as osp

ENV_NAME_TEMPLATE = (
    "mp3d-gibson-all-{task_type}-stage-2-{state_type}-state-final-run_*_*-{input_type}"
)

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
    'tasks/loopnav/mp3d.loopnav.yaml'
    --nav-task {task_type}
    --tensorboard-dir 'runs/{env_name}'
    --nav-env-verbose 0
""".replace(
    "\n", " "
)


def build_all_args():
    task_types = ["loopnav", "teleportnav"]
    state_types = ["trained", "zero", "random"]
    input_types = ["blind", "no-inputs"]

    parser = ModelEvaluator.build_parser()

    for task_type, state_type, input_type in itertools.product(
        task_types, state_types, input_types
    ):
        for ckpt_dir in glob.glob(
            osp.join(
                "data/checkpoints/probes",
                ENV_NAME_TEMPLATE.format(
                    task_type=task_type, state_type=state_type, input_type=input_type
                ),
            )
        ):
            yield parser.parse_args(
                args=shlex.split(
                    ARGS_STR_TEMPLATE.format(
                        task_type=task_type,
                        checkpoint_dir=ckpt_dir,
                        env_name=osp.basename(ckpt_dir),
                    )
                )
            )


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
    slurm_partition="learnfair",
    name="nav-analysis-eval",
    slurm_signal_delay_s=60,
)


jobs = []
with executor.batch():
    for args in build_all_args():
        jobs.append(executor.submit(ModelEvaluator(), args))


for job in jobs:
    print(job.results())
