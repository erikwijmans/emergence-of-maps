import getpass
import os
import shlex
import subprocess
import uuid

import filelock
import tinydb

from launchery.make_env_and_launch import (
    BASE_ENV,
    CONDA_ACTIVATE_CMD,
    ENV_DB,
    ENV_DB_LOCK,
    call,
    user,
)


def main():
    squeue = [
        l.strip()
        for l in call("squeue -u {}".format(user), capture_out=True)
        .decode("utf-8")
        .split("\n")
        if len(l.strip()) > 0
    ][1:]

    active_job_ids = []
    for job in squeue:
        jid = int(job.split(" ")[0].split("_")[0])
        active_job_ids.append(jid)

    active_job_ids = set(active_job_ids)

    with filelock.FileLock(ENV_DB_LOCK), tinydb.TinyDB(ENV_DB) as db:
        for row in db.all():
            if row["jid"] in active_job_ids:
                continue

            call("conda env remove --yes --name {}".format(row["env_name"]))

            CondaEnv = tinydb.Query()
            db.remove(CondaEnv.env_name == row["env_name"])


if __name__ == "__main__":
    main()
