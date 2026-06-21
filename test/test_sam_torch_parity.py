"""Parity tests for the batched PyTorch SAM dynamics (``SAM_torch.SAMTorch``).

The batched model must reproduce the scalar NumPy ``SAM.dynamics`` exactly (up to
float64 round-off), and the batched ``rollout_parallel`` on the SAM robot wrapper must
reproduce the per-(knot, sample, region) NumPy Euler rollout it replaces.

Run::

    pytest classes/robots/smarc_modelling/test/test_sam_torch_parity.py -v

Needs the ``admm`` conda env (numpy + torch).  No GCS solve / Mosek required.
"""
import pathlib
import sys

import numpy as np

# Repo root on sys.path (mirrors test_sam_rollout.py):
#   parents[0]=test [1]=smarc_modelling [2]=robots [3]=classes [4]=bundle-stl
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Put the smarc_modelling package src on path so the vehicle modules import.
_SMARC_SRC = REPO_ROOT / "classes" / "robots" / "smarc_modelling" / "src"
if str(_SMARC_SRC) not in sys.path:
    sys.path.insert(0, str(_SMARC_SRC))

from smarc_modelling.vehicles.SAM import SAM as SAMNumpy       # noqa: E402
from smarc_modelling.vehicles.SAM_torch import SAMTorch        # noqa: E402

DT = 0.02


def _random_states(n, rng):
    """A batch of physically-plausible 19-D SAM states with unit quaternions."""
    X = np.zeros((n, 19))
    X[:, 0:3] = rng.uniform(-2.0, 2.0, size=(n, 3))          # position
    q = rng.standard_normal((n, 4))                          # quaternion
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    X[:, 3:7] = q
    X[:, 7:13] = rng.uniform(-1.5, 1.5, size=(n, 6))         # body velocities
    X[:, 13] = rng.uniform(0.0, 100.0, size=n)               # x_vbs
    X[:, 14] = rng.uniform(0.0, 100.0, size=n)               # x_lcg
    X[:, 15] = rng.uniform(-0.12, 0.12, size=n)              # delta_s
    X[:, 16] = rng.uniform(-0.12, 0.12, size=n)              # delta_r
    X[:, 17] = rng.uniform(0.0, 1525.0, size=n)              # rpm1
    X[:, 18] = rng.uniform(0.0, 1525.0, size=n)              # rpm2
    return X


def _random_controls(n, rng):
    U = np.zeros((n, 6))
    U[:, 0] = rng.uniform(0.0, 100.0, size=n)                # x_vbs
    U[:, 1] = rng.uniform(0.0, 100.0, size=n)                # x_lcg
    U[:, 2] = rng.uniform(-0.12, 0.12, size=n)              # delta_s
    U[:, 3] = rng.uniform(-0.12, 0.12, size=n)              # delta_r
    U[:, 4] = rng.uniform(0.0, 1525.0, size=n)               # rpm1
    U[:, 5] = rng.uniform(0.0, 1525.0, size=n)               # rpm2
    return U


def test_dynamics_parity():
    """Row-by-row ``SAMTorch.dynamics`` == ``SAM.dynamics`` to < 1e-8."""
    rng = np.random.default_rng(0)
    n = 256
    X = _random_states(n, rng)
    U = _random_controls(n, rng)

    ref = SAMNumpy(dt=DT)
    dX_ref = np.stack([ref.dynamics(X[i].copy(), U[i].copy()) for i in range(n)])

    sam_t = SAMTorch(dt=DT, device="cpu")
    dX_t = sam_t.dynamics(X, U)

    err = np.max(np.abs(dX_ref - dX_t))
    assert err < 1e-8, f"max dynamics parity error {err:.3e} too large"


def test_dynamics_parity_zero_throttle():
    """Edge case: rpm exactly 0 (the n_rps>0 branch boundary)."""
    rng = np.random.default_rng(1)
    n = 64
    X = _random_states(n, rng)
    U = _random_controls(n, rng)
    U[:, 4:6] = 0.0                                          # both props off
    X[:, 17:19] = 0.0

    ref = SAMNumpy(dt=DT)
    dX_ref = np.stack([ref.dynamics(X[i].copy(), U[i].copy()) for i in range(n)])

    sam_t = SAMTorch(dt=DT, device="cpu")
    dX_t = sam_t.dynamics(X, U)

    err = np.max(np.abs(dX_ref - dX_t))
    assert err < 1e-8, f"max parity error (zero throttle) {err:.3e} too large"


def test_rollout_parallel_parity():
    """The batched ``rollout_parallel`` reproduces the sequential numpy Euler rollout.

    Compares the torch-backed ``W_F`` against the original triple-loop over the scalar
    ``_euler_rollout`` (kept as the numpy fallback), plus checks the residual arrays.
    """
    from classes.robots.sam import SAM as SAMRobot   # noqa: E402
    from classes.problem import Problem              # noqa: E402
    from utils.controls import Controls              # noqa: E402

    robot = SAMRobot(dt=0.6, n_euler=10, corridor_length=3.0)
    # Force the batched twin onto CPU so accumulated round-off stays at float64 level.
    robot._sam_torch = SAMTorch(dt=robot.dt / robot.n_euler, device="cpu")

    # Minimal Problem just to populate the residual weights.
    goal = np.zeros(robot.nx)
    goal[0] = robot.corridor_length
    problem = Problem(
        robot=robot, N=5, start_state=np.zeros(robot.nx), goal_state=goal,
        R_stage=2.0, Q_terminal=3.0, M_s=4, u_order=1,
    )
    robot.configure_residuals(problem)

    N, Ms, K = 5, 4, len(robot.regions)
    NX, NU, u_order = robot.nx, robot.nu, 1
    rng = np.random.default_rng(3)

    # Random in-corridor states with unit quaternions.
    W_X = np.zeros((N, Ms, K, NX))
    W_X[..., 0] = rng.uniform(0.0, robot.corridor_length, size=(N, Ms, K))
    W_X[..., 1] = rng.uniform(-0.5, 0.5, size=(N, Ms, K))
    W_X[..., 2] = robot.depth
    q = rng.standard_normal((N, Ms, K, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    W_X[..., 3:7] = q
    W_X[..., 7:13] = rng.uniform(-0.5, 0.5, size=(N, Ms, K, 6))
    W_X[..., 13] = 50.0
    W_X[..., 14] = 50.0

    # Random Controls (same object across regions, as the sampler produces).
    W_U = np.ndarray((N - 1, Ms, K), dtype=np.object_)
    for i in range(N - 1):
        for j in range(Ms):
            cps = rng.uniform(-1.0, 1.0, size=(NU, u_order + 1))
            ctrl = Controls(cps=cps)
            for k in range(K):
                W_U[i, j, k] = ctrl

    W_F, W_T, W_R, W_R_term = robot.rollout_parallel(W_X, W_U, robot.regions)

    # Reference forward states via the sequential numpy rollout.
    W_F_ref = robot._rollout_F_numpy(W_X, W_U, robot.dt, N - 1, Ms, K, NX)

    err = np.max(np.abs(W_F - W_F_ref))
    assert err < 1e-8, f"rollout W_F parity error {err:.3e} too large"

    assert np.all(W_T == robot.dt)
    # Stage residual cross-check on a couple of entries.
    assert np.allclose(W_R[0, 0, 0], robot._res_sqrt_R * W_U[0, 0, 0].cps.flatten())
    assert np.allclose(W_R_term[0, 0], robot._res_sqrt_Q * (W_X[N - 1, 0, 0, :2] - robot._res_goal))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = _random_states(256, rng)
    U = _random_controls(256, rng)
    ref = SAMNumpy(dt=DT)
    dX_ref = np.stack([ref.dynamics(X[i].copy(), U[i].copy()) for i in range(256)])
    for dev in (["cpu", "cuda"] if __import__("torch").cuda.is_available() else ["cpu"]):
        dX_t = SAMTorch(dt=DT, device=dev).dynamics(X, U)
        print(f"[{dev}] max |dX_torch - dX_numpy| = {np.max(np.abs(dX_ref - dX_t)):.3e}")
