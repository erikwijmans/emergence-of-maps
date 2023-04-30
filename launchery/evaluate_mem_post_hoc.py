import warnings
import shlex
import os.path as osp
import tqdm
import msgpack_numpy
import os

warnings.simplefilter(action="ignore", category=FutureWarning)

from nav_analysis.evaluate_ppo import eval_checkpoint, ModelEvaluator


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


def main():
    parser = ModelEvaluator.build_parser()
    env_name = "none"

    data = dict(mem_len=[], spl=[], success=[])
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
            stats_means, stats_episodes, trained_args, num_frames, args = eval_checkpoint(args, model)


            for _, v in stats_episodes.items():
                data["mem_len"].append(args.max_memory_length)
                data["spl"].append(v["spl"])
                data["success"].append(v["success"])

    with open("spl_vs_mem_len_post_hoc.msg", "wb") as f:
        msgpack_numpy.pack(data, f, use_bin_type=True)


if __name__ == "__main__":
    main()
