"""Parity tests for the batched PyTorch SAM dynamics (``SAM_torch.SAMTorch``).

``SAMTorch`` uses a reduced **15-D** state ``[eta(7) | nu(6) | act(2)=[vbs, lcg]]``,
while the vendored NumPy ``SAM.dynamics`` integrates the full **19-D** state.  Parity
is therefore checked on the shared 15 dims only: the dropped fin/rpm actuator-state
derivatives have no analogue in the torch model (the fins/rpm act through the command).
The batched ``rollout_parallel`` on the SAM robot wrapper must still reproduce the
per-(knot, sample, region) scalar Euler rollout it replaces.

Run::

    pytest classes/robots/smarc_modelling/test/test_sam_torch_parity.py -v

Needs the ``admm`` conda env (numpy + torch).  No GCS solve / Mosek required.
"""
import pathlib
import sys

import numpy as np
import pytest

# Repo root on sys.path (mirrors test_sam_rollout.py):
#   parents[0]=test [1]=smarc_modelling [2]=robots [3]=utils [4]=bundle-stl
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Put the smarc_modelling package src on path so the vehicle modules import.
_SMARC_SRC = REPO_ROOT / "utils" / "robots" / "smarc_modelling" / "src"
if str(_SMARC_SRC) not in sys.path:
    sys.path.insert(0, str(_SMARC_SRC))

from smarc_modelling.vehicles.SAM import SAM as SAMNumpy       # noqa: E402
from smarc_modelling.vehicles.SAM_torch import SAMTorch        # noqa: E402

DT = 0.02

# The torch model is 15-D; the NumPy reference is 19-D.  Compare on the shared dims.
NX_TORCH = 15


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
    dX_t = sam_t.dynamics(X[:, :NX_TORCH], U)

    err = np.max(np.abs(dX_ref[:, :NX_TORCH] - dX_t))
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
    dX_t = sam_t.dynamics(X[:, :NX_TORCH], U)

    err = np.max(np.abs(dX_ref[:, :NX_TORCH] - dX_t))
    assert err < 1e-8, f"max parity error (zero throttle) {err:.3e} too large"


def test_rollout_parallel_parity():
    """The batched ``rollout_parallel`` reproduces the scalar single-state rollout.

    Both go through the same ``SAMTorch`` model now, so this validates the pack/unpack and
    control-expansion of ``rollout_parallel`` against a per-triple ``robot.dynamics`` loop,
    plus checks the residual arrays.
    """
    from utils.robots.sam import SAM as SAMRobot   # noqa: E402
    from utils.problem import Problem              # noqa: E402
    from utils.controls import Controls              # noqa: E402

    robot = SAMRobot(dt=0.6, n_integrator=10, leg_length=3.0)
    # Force the batched twin onto CPU so accumulated round-off stays at float64 level.
    robot._sam_torch = SAMTorch(dt=robot.dt / robot.n_integrator, device="cpu")

    # Minimal Problem just to populate the residual weights.
    goal = np.zeros(robot.nx)
    goal[0] = robot.leg_length
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
    W_X[..., 0] = rng.uniform(0.0, robot.leg_length, size=(N, Ms, K))
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
            # [x_vbs%, x_lcg%, delta_s, delta_r, throttle] — VBS/LCG in a realistic
            # in-range band so the rollout exercises the buoyancy/CG dynamics.
            cps = rng.uniform(-1.0, 1.0, size=(NU, u_order + 1))
            cps[0] = rng.uniform(20.0, 80.0, size=u_order + 1)
            cps[1] = rng.uniform(20.0, 80.0, size=u_order + 1)
            ctrl = Controls(cps=cps)
            for k in range(K):
                W_U[i, j, k] = ctrl

    W_F, W_T, W_R, W_R_term = robot.rollout_parallel(W_X, W_U, robot.regions)

    # Reference forward states via the scalar single-state path (same torch model).
    W_F_ref = np.zeros((N - 1, Ms, K, NX))
    for i in range(N - 1):
        for j in range(Ms):
            for k in range(K):
                W_F_ref[i, j, k], _ = robot.dynamics(W_X[i, j, k], W_U[i, j, k], robot.dt)

    err = np.max(np.abs(W_F - W_F_ref))
    assert err < 1e-8, f"rollout W_F parity error {err:.3e} too large"

    assert np.all(W_T == robot.dt)
    # Stage residual cross-check on a couple of entries.
    assert np.allclose(W_R[0, 0, 0], robot._res_sqrt_R * W_U[0, 0, 0].cps.flatten())
    assert np.allclose(W_R_term[0, 0], robot._res_sqrt_Q * (W_X[N - 1, 0, 0, :2] - robot._res_goal))


# ---------------------------------------------------------------------------
# Learned damping (piml_type="pinn"): the optional state-dependent D path.
#
# The trained network is a build artefact, not source: `checkpoints/pinn.pt` is not
# in the repo.  Skip rather than fail so these do not mask the white-box parity
# gates above, which are what the dynamics rewrite is validated against.
# ---------------------------------------------------------------------------
from smarc_modelling.piml.pinn.pinn import _default_ckpt         # noqa: E402

requires_pinn = pytest.mark.skipif(
    not _default_ckpt().exists(),
    reason=f"PINN checkpoint not present (searched up to {_default_ckpt()})")


@requires_pinn
def test_pinn_D_matches_network():
    """SAMTorch's batched D equals the per-row reference ``pinn_D_predict``."""
    from smarc_modelling.piml.pinn.pinn import load_pinn_D, pinn_D_predict  # noqa: E402
    import torch                                                            # noqa: E402

    rng = np.random.default_rng(7)
    n = 64
    X = _random_states(n, rng)

    sam_t = SAMTorch(dt=DT, device="cpu", piml_type="pinn")
    feat = torch.cat([sam_t._t(X[:, 7:13]), sam_t._t(X[:, 13:19])], dim=1)
    with torch.no_grad():
        D_batch = sam_t.piml_model(feat).cpu().numpy()

    ref_net = load_pinn_D()
    err = max(np.max(np.abs(D_batch[i] - pinn_D_predict(ref_net, X[i, 7:13], X[i, 13:19])))
              for i in range(n))
    # Tolerance is the float32-network vs float64-dynamics rounding, not a logic gap.
    assert err < 1e-2, f"batched PINN D disagrees with reference by {err:.3e}"

    # D = A @ A.T must stay symmetric PSD for every row.
    for D in D_batch:
        assert np.max(np.abs(D - D.T)) < 1e-6
        assert np.linalg.eigvalsh(0.5 * (D + D.T)).min() > -1e-6


@requires_pinn
def test_pinn_equals_constant_at_zero_velocity():
    """With zero body velocity (and no current) ``D @ nu_r = 0`` for both models, so
    ``piml_type='pinn'`` and the default must give *bit-identical* dynamics — the flag
    changes nothing but the damping term."""
    rng = np.random.default_rng(8)
    n = 32
    X = _random_states(n, rng)[:, :NX_TORCH]
    X[:, 7:13] = 0.0                                   # nu = 0 -> nu_r = 0
    U = _random_controls(n, rng)

    d_const = SAMTorch(dt=DT, device="cpu", piml_type=None).dynamics(X, U)
    d_pinn = SAMTorch(dt=DT, device="cpu", piml_type="pinn").dynamics(X, U)
    err = np.max(np.abs(d_const - d_pinn))
    assert err < 1e-10, f"pinn/const dynamics differ at nu=0 by {err:.3e}"


@requires_pinn
def test_pinn_changes_damping_under_motion():
    """At nonzero velocity the learned D perturbs only the body-acceleration block;
    kinematics (eta_dot) and actuator dynamics are untouched."""
    rng = np.random.default_rng(9)
    n = 48
    X = _random_states(n, rng)[:, :NX_TORCH]           # nonzero body velocities
    U = _random_controls(n, rng)

    d_const = SAMTorch(dt=DT, device="cpu", piml_type=None).dynamics(X, U)
    d_pinn = SAMTorch(dt=DT, device="cpu", piml_type="pinn").dynamics(X, U)

    assert np.allclose(d_const[:, 0:7], d_pinn[:, 0:7]), "kinematics must be unchanged"
    assert np.allclose(d_const[:, 13:15], d_pinn[:, 13:15]), "actuator dyn unchanged"
    assert np.max(np.abs(d_const[:, 7:13] - d_pinn[:, 7:13])) > 1e-3, \
        "learned D should change nu_dot under motion"


@requires_pinn
def test_pinn_rollout_finite():
    """A batched Euler rollout with the learned D stays finite."""
    rng = np.random.default_rng(10)
    n = 16
    X0 = _random_states(n, rng)[:, :NX_TORCH]
    n_euler = 5
    U_traces = np.repeat(_random_controls(n, rng)[:, None, :], n_euler, axis=1)
    X = SAMTorch(dt=DT, device="cpu", piml_type="pinn").euler_rollout(
        X0, U_traces, duration=0.3, n_euler=n_euler)
    assert np.isfinite(X).all()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = _random_states(256, rng)
    U = _random_controls(256, rng)
    ref = SAMNumpy(dt=DT)
    dX_ref = np.stack([ref.dynamics(X[i].copy(), U[i].copy()) for i in range(256)])
    for dev in (["cpu", "cuda"] if __import__("torch").cuda.is_available() else ["cpu"]):
        dX_t = SAMTorch(dt=DT, device=dev).dynamics(X[:, :NX_TORCH], U)
        print(f"[{dev}] max |dX_torch - dX_numpy| = "
              f"{np.max(np.abs(dX_ref[:, :NX_TORCH] - dX_t)):.3e}")
