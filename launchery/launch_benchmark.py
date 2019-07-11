import os
import os.path as osp
import subprocess
import shlex


ngpus = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256][-1:]

MP3D = "tasks/mp3d-gibson.pointnav.yaml"
GIBSON = "tasks/gibson-public.pointnav.yaml"

for ngpu in ngpus:
    with open("./benchmark_etw.sh", "r") as f:
        template = f.read()

    params = dict(
        NGPU=ngpu,
        NTASKS=min(ngpu, 8),
        NNODES=(ngpu + 7) // 8,
        TASK=MP3D,
        FKEY="./data/perf_data/mp3d-gibson-sweep",
    )

    FKEY = params["FKEY"]
    if osp.exists(f"{FKEY}-{ngpu}.json"):
        os.remove(f"{FKEY}-{ngpu}.json")

    with open("/tmp/benchmark_batch.sh", "w") as f:
        f.write(template.format(**params))

    print(template.format(**params))

    subprocess.check_call(shlex.split("sbatch /tmp/benchmark_batch.sh"))
