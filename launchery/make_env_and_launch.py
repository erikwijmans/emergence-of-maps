import getpass
import os
import os.path as osp
import shlex
import subprocess
import datetime
import uuid
import argparse

user = getpass.getuser()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("script_name", type=str)
    parser.add_argument("--sbatch-opts", type=str, default="")

    return parser.parse_args()


def call(cmd, cwd=None, capture_out=False, env=None):
    cmd = shlex.split(cmd)

    if capture_out:
        return subprocess.check_output(cmd, cwd=cwd, env=env)
    else:
        subprocess.check_call(cmd, cwd=cwd, env=env)


def main():
    args = parse_args()

    with open(args.script_name, "r") as f:
        script_content = f.read()

    time = datetime.datetime.now()

    code_dir = osp.join(
        os.getcwd(),
        "sandbox",
        "code",
        "{}-{}-{}".format(time.year, time.month, time.day),
        "{}-{}-{}".format(time.hour, time.minute, time.second),
    )

    for cwd in [os.getcwd(), osp.join(os.getcwd(), "habitat-api-navigation-analysis")]:
        call(
            "python setup.py build --force --build-lib {}".format(code_dir),
            cwd=cwd,
            capture_out=True,
        )

    _id = uuid.uuid4()
    launch_script_name = "/tmp/{}.sh".format(_id)
    with open(launch_script_name, "w") as f:
        f.write(script_content)

    env_for_job = {k: v for k, v in os.environ.items()}
    env_for_job["PYTHONPATH"] = code_dir
    env_for_job = {k: v for k, v in env_for_job.items() if not k.startswith("SLURM_")}

    sbatch_res = call(
        "sbatch {} {}".format(args.sbatch_opts, launch_script_name),
        cwd=osp.join(os.getcwd(), "sandbox"),
        env=env_for_job,
    )


if __name__ == "__main__":
    main()
