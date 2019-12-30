import atexit
import collections
import copy
import json
import logging
import os
import os.path as osp
import random
import shlex
import signal
import socket
import subprocess
import sys
import threading

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from pydash import py_

from habitat import logger
from nav_analysis.benchmark_distrib_work import main as work_main
from nav_analysis.rl.ppo.utils import (
    batch_obs,
    ppo_args,
    update_linear_schedule,
)

logger.handlers[-1].setLevel(level=logging.WARNING)

INTERRUPTED = threading.Event()
INTERRUPTED.clear()

REQUEUE = threading.Event()
REQUEUE.clear()


def requeue_handler(signum, frame):
    # define the handler function
    # note that this is not executed here, but rather
    # when the associated signal is sent
    print("signaled for requeue")
    INTERRUPTED.set()
    REQUEUE.set()


signal.signal(signal.SIGUSR1, requeue_handler)


def init_distrib():
    master_port = int(os.environ.get("MASTER_PORT", 1234))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID")))
    world_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS")))

    torch.cuda.set_device(torch.device("cuda", local_rank))

    tcp_store = dist.TCPStore(master_addr, master_port, world_size, world_rank == 0)
    dist.init_process_group(
        dist.Backend.NCCL, store=tcp_store, rank=world_rank, world_size=world_size
    )

    return local_rank, tcp_store


def main():
    TASK = sys.argv[1]
    FKEY = sys.argv[2]
    sys.argv = sys.argv[0:1]
    n_repeats = 10
    sync_fracs = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

    local_rank, tcp_store = init_distrib()

    world_rank = dist.get_rank()
    ngpu = dist.get_world_size()

    random.seed(0)
    seeds = [random.randint(0, int(1e5)) for _ in range(n_repeats)]

    is_done_store = dist.PrefixStore("rollout_tracker", tcp_store)

    results = []
    outname = f"{FKEY}-{ngpu}.json"
    if osp.exists(outname):
        with open(outname, "r") as f:
            results = json.load(f)

    results_futures = []
    for sync_frac in sync_fracs:
        for i in range(n_repeats):
            args = ppo_args()
            args.ddppo.sync_frac = sync_frac
            args.general.seed = seeds[i]
            args.general.backprop = True
            args.ppo.num_updates = 10
            args.task.task_config = TASK
            args.general.ngpu = ngpu
            args.general.local_rank = local_rank

            if not py_.some(
                results, lambda v: v["sync_frac"] == sync_frac and v["seed"] == seeds[i]
            ):
                results_futures.append((args, is_done_store))

    for args in tqdm.tqdm(results_futures) if world_rank == 0 else results_futures:
        res = work_main(*args)
        results.append(res)
        if world_rank == 0:
            with open(f"{FKEY}-{ngpu}.json", "w") as f:
                json.dump(results, f)

            logger.warn(res)

        if INTERRUPTED.is_set():
            break

    if world_rank == 0:
        for sync_frac in sync_fracs:
            print(
                sync_frac,
                ngpu,
                py_()
                .filter(lambda v: v["sync_frac"] == sync_frac and v["ngpu"] == ngpu)
                .map("fps")
                .mean()(results),
            )

    if REQUEUE.is_set() and world_rank == 0:
        import time

        time.sleep(1)
        print("requeuing job " + os.environ["SLURM_JOB_ID"])
        os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])


if __name__ == "__main__":
    main()
