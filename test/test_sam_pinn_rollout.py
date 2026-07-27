"""PINN damping-matrix model vs. the white-box SAM model.

What this checks
----------------
SAM's body-frame damping ``D`` (the ``-D @ nu_r`` term in ``M nu_dot = tau - C nu_r
- D nu_r - g``) can be supplied two ways by ``SAM_PIML``:

* ``piml_type=None``  — the *white-box* / "normal" SAM model.  Its ``calculate_D``
  ends up overwriting ``D`` with a constant diagonal ``diag(60,60,60,5,5,5)``
  (``damping_factor`` / ``damping_rot``); it does not vary with the state.
* ``piml_type="pinn"`` — a Physics-Informed NN that *predicts* a full 6x6 ``D``
  from the body velocity and actuators.  The network outputs a flat 6x6 ``A`` and
  returns ``D = A @ A.T``, so ``D`` is symmetric positive-semidefinite by design
  and changes with the state (notably much lower surge damping than the constant
  white-box value, so the AUV coasts further).

The damping network shipped in ``checkpoints/pinn.pt`` is a *newer* format than the
``PINN`` class used by the training scripts (12-d ``concat(nu, u)`` input with
min/max input normalisation, a deeper trunk + a separate output layer, and
dropout).  It is loaded via ``load_pinn_D`` / ``pinn_D_predict`` (see
``piml/pinn/pinn.py``); this test is also the executable spec for that loader.

Run as a script for the readable summary::

    python classes/robots/smarc_modelling/test/test_sam_pinn_rollout.py

or under pytest::

    pytest classes/robots/smarc_modelling/test/test_sam_pinn_rollout.py -v

Needs the ``admm`` conda env (numpy + a CPU torch).  No ROS / rosbag, no GCS solve.
"""
import pathlib
import sys

import numpy as np

# This file lives inside the smarc_modelling submodule.  Put the repo root and the
# smarc_modelling package src on sys.path so imports resolve regardless of cwd
# (mirrors test_sam_rollout.py / test_sam_torch_parity.py):
#   parents[0]=test [1]=smarc_modelling [2]=robots [3]=utils [4]=bundle-stl
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_SMARC_SRC = REPO_ROOT / "utils" / "robots" / "smarc_modelling" / "src"
for _p in (REPO_ROOT, _SMARC_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest                                                   # noqa: E402

from smarc_modelling.vehicles.SAM_PIML import SAM_PIML          # noqa: E402
from smarc_modelling.piml.pinn.pinn import (                    # noqa: E402
    _default_ckpt, load_pinn_D, pinn_D_predict)

# The trained network is a build artefact, not source (`checkpoints/` is git-ignored), so
# it may be absent.  Skip rather than fail so these do not mask the white-box parity
# gates.  `_default_ckpt` searches $SAM_PINN_CKPT, checkpoints/ and the parent repo's
# assets/ — the same resolution the model itself uses.
pytestmark = pytest.mark.skipif(
    not _default_ckpt().exists(),
    reason=f"PINN checkpoint not present (searched up to {_default_ckpt()})")

# Constant the white-box branch collapses to (damping_factor / damping_rot).
WHITEBOX_D_DIAG = np.array([60.0, 60.0, 60.0, 5.0, 5.0, 5.0])


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def _make_model(piml_type, dt=0.1):
    return SAM_PIML(dt=dt, piml_type=piml_type)


def _rest_state():
    """Level, at rest, VBS/LCG half-filled.  Full 19-d state [eta, nu, u]."""
    return np.array([0.0, 0.0, 0.0,           # position
                     1.0, 0.0, 0.0, 0.0,      # quaternion (level)
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # nu
                     50.0, 50.0, 0.0, 0.0, 0.0, 0.0])  # u [vbs, lcg, ds, dr, rpm1, rpm2]


def _damping(model: SAM_PIML, eta, nu, u) -> np.ndarray:
    """Evaluate the model's 6x6 damping matrix at a given (eta, nu, u).

    Runs the same pre-step bookkeeping ``dynamics`` does so that the white-box
    branch sees a consistent ``nu_r`` before ``calculate_D``.
    """
    model.calculate_system_state(nu, eta, u)
    model.calculate_cg()
    model.update_inertias()
    model.calculate_D(eta, nu, u)
    return np.array(model.D)


def _rk4(f, x, u, dt):
    """RK4 step, identical to the integrator in piml_sim.SIM."""
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _rollout(model: SAM_PIML, x0, u_ref, dt=0.1, steps=30):
    x = x0.copy()
    traj = [x.copy()]
    for _ in range(steps):
        x = _rk4(model.dynamics, x, u_ref, dt)
        traj.append(x.copy())
    return np.array(traj)


# Representative (nu, u) operating points for the D comparison.
def _sample_points(n=25, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n):
        nu = rng.uniform(-1.0, 1.0, 6) * np.array([1.0, 0.5, 0.5, 0.5, 0.5, 1.0])
        u = np.array([rng.uniform(0, 100), rng.uniform(0, 100),
                      rng.uniform(-0.12, 0.12), rng.uniform(-0.10, 0.10),
                      rng.uniform(-1000, 1000), rng.uniform(-1000, 1000)])
        pts.append((nu, u))
    return pts


# ---------------------------------------------------------------------------
# 1. The loader / checkpoint contract
# ---------------------------------------------------------------------------
def test_pinn_loads_with_expected_io():
    """The checkpoint loads and exposes a 12-in / 6x6-out damping map."""
    model = load_pinn_D()
    assert model.layers[0].in_features == 12       # concat(nu[6], u[6])
    assert model.output_layer.out_features == 36    # flat 6x6 A
    D = pinn_D_predict(model, np.zeros(6), np.array([50, 50, 0, 0, 0, 0.0]))
    assert D.shape == (6, 6)
    assert np.isfinite(D).all()


def test_pinn_D_symmetric_psd_finite():
    """D = A @ A.T must be symmetric, PSD and finite at every operating point."""
    model = load_pinn_D()
    for nu, u in _sample_points():
        D = pinn_D_predict(model, nu, u)
        assert np.isfinite(D).all()
        assert np.max(np.abs(D - D.T)) < 1e-4, "D not symmetric"
        eig_min = np.linalg.eigvalsh(0.5 * (D + D.T)).min()
        assert eig_min > -1e-5, f"D not PSD (min eig {eig_min:.2e})"


# ---------------------------------------------------------------------------
# 2. PINN vs white-box damping
# ---------------------------------------------------------------------------
def test_whitebox_D_is_constant_diagonal():
    """Sanity-anchor the baseline: the white-box D is the fixed diagonal."""
    wb = _make_model(None)
    eta = _rest_state()[0:7]
    D_a = _damping(wb, eta, np.zeros(6), np.array([50, 50, 0, 0, 0, 0.0]))
    D_b = _damping(wb, eta, np.array([1.0, 0.3, 0.0, 0, 0, 0.2]),
                   np.array([50, 50, 0, 0, 500, 500.0]))
    assert np.allclose(np.diag(D_a), WHITEBOX_D_DIAG)
    assert np.allclose(D_a, D_b), "white-box D should not depend on the state"
    assert np.allclose(D_a, np.diag(np.diag(D_a))), "white-box D should be diagonal"


def test_pinn_D_differs_from_whitebox():
    """The learned D is a full matrix, materially different from the constant."""
    wb, pinn = _make_model(None), _make_model("pinn")
    eta = _rest_state()[0:7]
    nu = np.array([0.6, 0.1, 0.0, 0.0, 0.0, 0.3])
    u = np.array([50, 50, 0, 0, 400, 400.0])

    D_wb = _damping(wb, eta, nu, u)
    D_pinn = _damping(pinn, eta, nu, u)

    # Off-diagonal coupling the diagonal white-box model cannot represent.
    off_diag = D_pinn - np.diag(np.diag(D_pinn))
    assert np.max(np.abs(off_diag)) > 1.0, "PINN D has no real cross-coupling"
    # And it is not just the white-box diagonal in disguise.
    assert np.linalg.norm(D_pinn - D_wb) > 1.0


def test_pinn_D_is_velocity_dependent():
    """Unlike the white-box constant, the PINN D changes with body velocity."""
    pinn = _make_model("pinn")
    eta = _rest_state()[0:7]
    u = np.array([50, 50, 0, 0, 300, 300.0])
    D_slow = _damping(pinn, eta, np.zeros(6), u)
    D_fast = _damping(pinn, eta, np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0]), u)
    assert np.linalg.norm(D_fast - D_slow) > 1e-3


# ---------------------------------------------------------------------------
# 3. Rollout comparison
# ---------------------------------------------------------------------------
def test_rollouts_finite():
    """Both models integrate to a finite trajectory under steady thrust."""
    u_ref = np.array([50, 50, 0, 0, 500, 500.0])
    x0 = _rest_state()
    for kind in (None, "pinn"):
        traj = _rollout(_make_model(kind), x0, u_ref)
        assert np.isfinite(traj).all(), f"{kind} rollout diverged"


def test_pinn_coasts_further_than_whitebox():
    """The PINN's low surge damping means more forward travel than the heavily
    over-damped white-box model under identical thrust."""
    u_ref = np.array([50, 50, 0, 0, 500, 500.0])
    x0 = _rest_state()
    x_wb = _rollout(_make_model(None), x0, u_ref)[-1, 0]
    x_pinn = _rollout(_make_model("pinn"), x0, u_ref)[-1, 0]
    assert x_pinn > x_wb, f"PINN x={x_pinn:.3f} should exceed white-box x={x_wb:.3f}"


# ---------------------------------------------------------------------------
# Runnable diagnostic: `python .../test_sam_pinn_rollout.py` prints the summary.
# ---------------------------------------------------------------------------
def _print_summary():
    wb, pinn = _make_model(None), _make_model("pinn")
    eta = _rest_state()[0:7]
    u = np.array([50, 50, 0, 0, 400, 400.0])

    print("=" * 72)
    print("SAM damping: white-box (piml_type=None) vs PINN (piml_type='pinn')")
    print("=" * 72)

    np.set_printoptions(precision=2, suppress=True)
    for label, nu in [("at rest", np.zeros(6)),
                      ("surge=1.0 m/s", np.array([1.0, 0, 0, 0, 0, 0.0]))]:
        D_wb = _damping(wb, eta, nu, u)
        D_pinn = _damping(pinn, eta, nu, u)
        print(f"\n-- D, {label} --")
        print("  white-box diag :", np.diag(D_wb))
        print("  PINN      diag :", np.diag(D_pinn))
        print("  PINN max |off-diag| :", np.max(np.abs(D_pinn - np.diag(np.diag(D_pinn)))).round(2))
        print("  ||D_pinn - D_wb||_F :", round(float(np.linalg.norm(D_pinn - D_wb)), 2))
        print("  PINN eig(min,max)   :",
              tuple(np.round([np.linalg.eigvalsh(0.5 * (D_pinn + D_pinn.T)).min(),
                              np.linalg.eigvalsh(0.5 * (D_pinn + D_pinn.T)).max()], 2)))

    u_ref = np.array([50, 50, 0, 0, 500, 500.0])
    x0 = _rest_state()
    end_wb = _rollout(wb, x0, u_ref)[-1]
    end_pinn = _rollout(pinn, x0, u_ref)[-1]
    print("\n-- 3 s rollout under steady thrust [rpm=500] --")
    print("  white-box  final pos:", end_wb[:3].round(3), " nu:", end_wb[7:13].round(3))
    print("  PINN       final pos:", end_pinn[:3].round(3), " nu:", end_pinn[7:13].round(3))
    print("  => PINN's lower surge damping lets the AUV coast further forward.")
    print("=" * 72)


if __name__ == "__main__":
    _print_summary()
