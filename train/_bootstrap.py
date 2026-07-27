"""Make both import roots available to the ``train/`` scripts.

``train/`` sits at the SUBMODULE ROOT (sibling of ``src/``, ``test/``, ``data/``,
``checkpoints/``), so it is not importable as ``smarc_modelling.train`` and it is not
covered by the parent repo's ``conftest.py``.  Every entry point here starts with
``import _bootstrap`` (or ``from . import _bootstrap``) to get the same two roots the
test suite relies on:

* the parent repo root, so ``import utils.robots.sam`` and
  ``import benchmarking.sam_rollout.bench_rollout`` resolve;
* ``<submodule>/src``, so ``import smarc_modelling.vehicles.SAM_torch`` resolves.

Mirrors /home/none/gits/PROJECTS/bundle-stl/conftest.py and
benchmarking/sam_rollout/bench_rollout.py.
"""
import pathlib
import sys

#: parents[0]=train [1]=<submodule root> [2]=robots [3]=utils [4]=<parent repo root>
_HERE = pathlib.Path(__file__).resolve()
SUBMODULE_ROOT = _HERE.parents[1]
REPO_ROOT = _HERE.parents[4]
SMARC_SRC = SUBMODULE_ROOT / "src"

DATA_DIR = SUBMODULE_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CHECKPOINT_DIR = SUBMODULE_ROOT / "checkpoints"
TRAIN_DIR = SUBMODULE_ROOT / "train"
RESULTS_DIR = TRAIN_DIR / "results"

for _p in (REPO_ROOT, SMARC_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
