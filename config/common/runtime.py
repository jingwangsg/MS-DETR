import os.path as osp
from pathlib import Path

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../.."))

paths = dict(root_dir=ROOT_DIR,
             data_dir=osp.join(ROOT_DIR, "data-bin", "raw"),
             cache_dir=osp.join(ROOT_DIR, "data-bin", "cache"),
             work_dir=osp.join(ROOT_DIR, "work_dir", "${data.dataset}", "${flags.exp}"))

flags = dict(debug=False, ddp=False, amp=False, train=True, wandb=False, seed=3147)
