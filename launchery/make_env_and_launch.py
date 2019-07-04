import getpass
import os
import os.path as osp
import shlex
import subprocess
import datetime
import uuid

user = getpass.getuser()


def call(cmd, cwd=None, capture_out=False, env=None):
    cmd = shlex.split(cmd)

    if capture_out:
        return subprocess.check_output(cmd, cwd=cwd, env=env)
    else:
        subprocess.check_call(cmd, cwd=cwd, env=env)


def main():
    if user == "erikwijmans":
        script_name = "distrib_batch_etw.sh"
    elif user == "akadian":
        script_name = "distrib_batch_kadian.sh"

    with open(script_name, "r") as f:
        script_content = f.read()

    time = datetime.datetime.now()

    code_dir = osp.join(
        os.getcwd(),
        "sandbox",
        "code",
        "{}-{}-{}".format(time.year, time.month, time.day),
        "{}-{}-{}".format(time.hour, time.minute, time.second),
    )

    for cwd in [
        os.getcwd(),
        osp.join(os.getcwd(), "habitat-api-navigation-analysis"),
    ]:
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

    sbatch_res = call(
        "sbatch {}".format(launch_script_name),
        cwd=osp.join(os.getcwd(), "sandbox"),
        env=env_for_job,
    )

    os.remove(launch_script_name)


if __name__ == "__main__":
    main()
