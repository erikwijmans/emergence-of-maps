import os
import os.path as osp
from typing import List

install_paths: List[str] = [
    os.getcwd(),
    osp.join(os.getcwd(), "habitat-api-navigation-analysis"),
]
