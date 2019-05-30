import getpass
import os
import os.path as osp
import shlex
import subprocess
import uuid

import filelock
import tinydb

ENV_DB = osp.abspath(osp.join(osp.dirname(__file__), "envs_for_jobs.json"))
ENV_DB_LOCK = osp.abspath(
    osp.join(osp.dirname(__file__), "envs_for_jobs.json.lock")
)

user = getpass.getuser()
assert user in ["erikwijmans", "akadian"]

if user == "erikwijmans":
    BASE_ENV = "nav-analysis-base"
    CONDA_ACTIVATE_CMD = (
        "source /private/home/erikwijmans/miniconda3/etc/profile.d/conda.sh"
    )


def call(cmd, cwd=None, capture_out=False, env=None):
    cmd = "{} && {}".format(CONDA_ACTIVATE_CMD, cmd)
    cmd = 'bash -c "{}"'.format(cmd)
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

    _id = uuid.uuid4()
    env_name = "nav-analysis-{}".format(_id)

    call("conda create --name {} --clone {}".format(env_name, BASE_ENV))
    call("conda deactivate")
    call("conda activate {} && pip install .".format(env_name))
    call(
        "conda activate {} && pip install .".format(env_name),
        cwd=os.path.join(os.getcwd(), "habitat-api-navigation-analysis"),
    )

    launch_script_name = "/tmp/{}.sh".format(_id)
    with open(launch_script_name, "w") as f:
        f.write(script_content)

    env_for_job = {
        k: v for k, v in os.environ.items() if k not in {"PYTHONPATH"}
    }

    sbatch_res = call(
        "sbatch {} {}".format(launch_script_name, env_name),
        capture_out=True,
        cwd=os.path.join(os.getcwd(), "sandbox"),
        env=env_for_job,
    )
    sbatch_res = sbatch_res.decode("utf-8").strip()
    print(sbatch_res)
    jid = int(sbatch_res.split(" ")[-1])

    with filelock.FileLock(ENV_DB_LOCK), tinydb.TinyDB(ENV_DB) as db:
        db.insert(dict(jid=jid, env_name=env_name))

    os.remove(launch_script_name)


if __name__ == "__main__":
    main()
