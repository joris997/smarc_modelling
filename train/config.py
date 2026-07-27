"""Every training hyperparameter in one dataclass, dumped beside the checkpoint."""
import dataclasses
import os
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainConfig:
    # --- reproducibility ---------------------------------------------------
    seed: int = 0
    deterministic: bool = True
    device: str = "auto"
    dtype: str = "float32"        # the net trains and ships fp32; fp64 costs 16x on sm_89

    # --- architecture ------------------------------------------------------
    hidden: int = 32
    n_hidden: int = 2
    activation: str = "tanh"
    fossen_split: bool = True

    # --- data --------------------------------------------------------------
    use_bad_bags: bool = True
    bad_bag_weight: float = 0.25
    min_speed: float = 0.02       # samples with ||nu|| below this carry no damping signal

    # --- stage A: one-step pretrain ----------------------------------------
    # Demoted deliberately.  Least squares on this target reaches R^2 = -0.071 for the best
    # constant FULL D, and the white-box initialisation is already a better starting point
    # than Stage A can produce -- the residual is white-box thrust/buoyancy error that no PD
    # D can represent.  Kept because it is the signal the previous model was trained on and
    # so makes the comparison like-for-like; `--skip-stage-a` turns it off.
    stage_a_epochs: int = 300
    stage_a_lr: float = 3e-3
    stage_a_w_data: float = 1.0
    stage_a_w_anchor: float = 1e-2
    stage_a_w_stiff: float = 1e-3

    # --- stage B: multi-step rollout fine-tune -----------------------------
    stage_b_epochs: int = 400
    stage_b_lr: float = 3e-4
    stage_b_w_roll: float = 1.0
    stage_b_w_data: float = 0.1
    stage_b_w_anchor: float = 1e-3
    stage_b_w_stiff: float = 1e-2
    # Wider batches, fewer of them: the rollout is launch-bound (each epoch issues
    # batches x H x n_sub x 4 separate eager `_dyn` calls), so at b=256 the GPU idles
    # between kernels.  b=1024 x 5 does 2x the samples per epoch in 4x fewer launches.
    batch_windows: int = 1024
    batches_per_epoch: int = 5
    horizon_schedule: tuple = (2, 4, 8, 16)   # curriculum, split evenly over the epochs
    gamma: float = 0.9                        # per-step discount inside a window
    huber_delta: float = 2.0
    n_sub: int = 2                            # RK4 substeps per data interval
    integrator: str = "rk4"

    # --- shared ------------------------------------------------------------
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_frac: float = 0.05
    patience: int = 40
    stiff_target: float = 60.0                # 1/s; hinge knee for lambda_max(M^-1 dDnu/dnu)
    eval_every: int = 10
    #: Rows used per step for the anchor / stiffness averages.  Evaluating them over the
    #: full ~4.4k training set every minibatch dominated step time (batched `eigvalsh`);
    #: a fresh random subset is an unbiased estimate for a fraction of the cost.
    reg_subsample: int = 512

    # --- output ------------------------------------------------------------
    out_name: str = "pinn_reduced.pt"
    tag: str = ""

    def horizon_for(self, epoch, total):
        """Curriculum lookup: which rollout horizon this epoch trains at."""
        k = len(self.horizon_schedule)
        return self.horizon_schedule[min(k - 1, int(k * epoch / max(total, 1)))]

    def to_dict(self):
        d = dataclasses.asdict(self)
        d["horizon_schedule"] = list(self.horizon_schedule)
        return d


def seed_everything(cfg):
    """Seed every RNG and return a dedicated generator for the window sampler."""
    import torch
    if cfg.deterministic:
        # Required by torch for deterministic cuBLAS reductions; must be set before the
        # first CUDA matmul, hence here rather than in the shell.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    return np.random.default_rng(cfg.seed)


def resolve_device(cfg):
    import torch
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
