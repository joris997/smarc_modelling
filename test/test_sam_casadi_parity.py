"""Parity: SAM_casadi_bundle.SAMCasadiBundle vs SAM_torch.SAMTorch (the bundle's
source of truth).  Every conclusion drawn from the CasADi NLP rests on this test —
if the two models disagree, the NLP is answering a different question.

Run:  pytest classes/robots/smarc_modelling/test/test_sam_casadi_parity.py -q
"""
import os
import sys

import numpy as np
import pytest

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from smarc_modelling.vehicles.SAM_torch import SAMTorch
from smarc_modelling.vehicles.SAM_casadi_bundle import SAMCasadiBundle

DT_SUB = 2.5 / 50.0      # the substep sam.py actually uses (knot dt / n_integrator)


def _rand_states(rng, b):
    X = np.zeros((b, 15))
    X[:, 0:3] = rng.uniform(-3, 3, (b, 3))
    q = rng.normal(size=(b, 4))
    X[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    X[:, 7:10] = rng.uniform(-0.5, 0.5, (b, 3))     # u, v, w
    X[:, 10:13] = rng.uniform(-0.5, 0.5, (b, 3))    # p, q, r
    X[:, 13:15] = rng.uniform(0, 100, (b, 2))       # vbs, lcg
    return X


def _rand_urefs(rng, b):
    U = np.zeros((b, 6))
    U[:, 0:2] = rng.uniform(0, 100, (b, 2))         # vbs, lcg cmd
    U[:, 2:4] = rng.uniform(-0.3, 0.3, (b, 2))      # delta_s, delta_r
    rpm = rng.uniform(-500, 500, b)                 # SIGNED rpm (bipolar throttle)
    U[:, 4] = U[:, 5] = rpm
    return U


@pytest.fixture(scope="module")
def models():
    return SAMTorch(dt=DT_SUB, device="cpu"), SAMCasadiBundle(dt=DT_SUB, smooth=False)


def test_dynamics_parity(models):
    """dx = f(x, u) must agree to ~machine precision on random points."""
    torch_m, cas_m = models
    rng = np.random.default_rng(0)
    X, U = _rand_states(rng, 64), _rand_urefs(rng, 64)

    dX_t = np.asarray(torch_m.dynamics(X, U))
    dX_c = np.stack([np.asarray(cas_m.f(X[i], U[i])).ravel() for i in range(X.shape[0])])

    err = np.abs(dX_t - dX_c)
    worst_dim = int(np.argmax(err.max(axis=0)))
    assert err.max() < 1e-9, (
        f"casadi/torch dynamics diverge: max|dx| err = {err.max():.3e} "
        f"(worst state dim {worst_dim}); per-dim max = {err.max(axis=0)}")


def test_rollout_parity(models):
    """A full RK4 knot rollout (50 substeps over dt=2.5) must agree."""
    torch_m, cas_m = models
    rng = np.random.default_rng(1)
    b, n_sub, dur = 8, 50, 2.5
    X, U = _rand_states(rng, b), _rand_urefs(rng, b)

    # torch: rk4_rollout wants a per-substep control trace (b, n_sub, 6)
    U_trace = np.repeat(U[:, None, :], n_sub, axis=1)
    Xn_t = np.asarray(torch_m.rk4_rollout(X, U_trace, dur, n_sub))
    Xn_c = np.stack([np.asarray(cas_m.rollout(X[i], U[i], dur, n_sub)).ravel()
                     for i in range(b)])

    err = np.abs(Xn_t - Xn_c)
    assert err.max() < 1e-8, (
        f"casadi/torch rollout diverge: max err = {err.max():.3e}; "
        f"per-dim max = {err.max(axis=0)}")


def test_expand_control_matches_robot():
    """The CasADi throttle map must match SAM._expand_control_batch bit-for-bit."""
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
    from utils.robots.sam import SAM
    import casadi as cs

    robot = SAM(dt=2.5, n_integrator=5, scenario="L", piml_type=None, integrator="rk4")
    for s in np.linspace(-1, 1, 21):
        # The robot's 5-D control carries VBS/LCG NORMALISED (SAM.ACT_SCALE) and
        # `_expand_control_batch` converts them to percent; the CasADi mirror takes
        # percent directly.  Feed each its own units -- the map under test is the
        # THROTTLE (dims 4,5), which is unaffected by the actuator rescale.
        c5_norm = np.array([0.5, 0.5, 0.05, -0.05, s])
        c5_pct = np.array([50.0, 50.0, 0.05, -0.05, s])
        ref = robot._expand_control_batch(c5_norm[None])[0]
        got = np.asarray(SAMCasadiBundle.expand_control(
            cs.DM(c5_pct), robot.rpm_max, robot.rpm_rev_max,
            robot.rev_thrust_ratio)).ravel()
        assert np.abs(ref - got).max() < 1e-9, f"throttle map differs at s={s}: {ref} vs {got}"
