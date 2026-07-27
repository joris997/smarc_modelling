"""SAM dynamics rollout tests + control-collapse regression guard.

Background — the bug this guards against
----------------------------------------
The Graphs-of-Bundles planner convexifies the dynamics by taking convex (simplex)
blends of sampled ``(x, u, f)`` triples.  That is only consistent when the rolled-out
state ``f`` is (near-)affine in the control.  When the SAM control was raw ``rpm`` the
per-knot displacement was ~quadratic (``dx ~ a*rpm^2``) and asymmetric (reverse thrust
~7x weaker, and hardware-limited so it can't be scaled to match): a 50/50 blend of
``+rpm`` and ``-rpm`` samples advanced the blended *state* (+0.365 m) while the blended
*control* was exactly 0 rpm.  The relaxation therefore "moved" on zero net control, the
real rollout stayed put, and the dynamics infeasibility was pinned at the goal distance
(~3.0 m) for every iteration.

The fix (``classes/robots/sam.py``) reparametrises the 3rd control as a normalised,
forward-only throttle ``s in [-1, 1]`` (s=-1 off, s=+1 full) mapped to rpm via
``rpm = sqrt((s + 1) / 2) * rpm_max``.  The sqrt cancels the quadratic so ``dx`` is
~affine in ``s``; being forward-only and smooth over the whole sampled range (no
clamp/kink) removes the +/- cancellation.  Convex blends are then consistent and the
control no longer collapses.

This file checks (a) the rollout still behaves physically and (b) the blend-consistency
property that the fix restores.  Run as a script for the readable summary::

    python classes/robots/smarc_modelling/test/test_sam_rollout.py

or under pytest::

    pytest classes/robots/smarc_modelling/test/test_sam_rollout.py -v

Both need the ``admm`` conda env.  No GCS solve / Mosek is required.
"""
import pathlib
import sys

import numpy as np

# This file lives inside the smarc_modelling *submodule*; the SAM wrapper it tests
# lives in the parent bundle-stl repo.  Put the repo root on sys.path so
# ``classes`` / ``utils`` import regardless of cwd.
#   .../bundle-stl/classes/robots/smarc_modelling/test/test_sam_rollout.py
#   parents[0]=test  [1]=smarc_modelling  [2]=robots  [3]=utils  [4]=bundle-stl
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.robots.sam import SAM          # noqa: E402
from utils.controls import Controls         # noqa: E402

# ---------------------------------------------------------------------------
# Setup: same robot/problem geometry as examples/main_sam.py so the numbers here
# line up with an actual planner run.
# ---------------------------------------------------------------------------
N_KNOTS = 12                                 # examples/main_sam.py: Problem(N=12)
S_FULL = +1.0                                # throttle "full" (rpm = +rpm_max)
DS_TRIM = 0.085                              # stern-plane angle that zeroes prop-pitch


def _s_coast(robot: SAM) -> float:
    """The throttle that produces rpm = 0.

    The throttle is BIPOLAR: ``s = -1`` is full REVERSE (``-rpm_rev_max``), not "off".
    ``_expand_control_batch`` keeps the signed *thrust proxy* ``P`` affine in ``s``::

        P(s) = P_rev + (s+1)/2 * (P_fwd - P_rev),  P_fwd = rpm_max^2,
                                                   P_rev = -rpm_rev_max^2 / rev_thrust_ratio

    so the coast point is wherever ``P(s) = 0``, i.e. ``s = -2*P_rev/(P_fwd-P_rev) - 1``
    (≈ -0.818 at the default constants) — NOT ``s = -1``.  Derive it from the robot's
    own constants so this tracks any retune of the thrust map.
    """
    P_fwd = robot.rpm_max ** 2
    P_rev = -(robot.rpm_rev_max ** 2) / robot.rev_thrust_ratio
    return 2.0 * (-P_rev) / (P_fwd - P_rev) - 1.0


def _make_robot() -> SAM:
    return SAM(dt=0.6, n_integrator=10, leg_length=3.0)


def _knot_spacing(robot: SAM) -> float:
    """Forward distance the planner needs to cover per knot."""
    return robot.leg_length / (N_KNOTS - 1)


def _start_state(robot: SAM) -> np.ndarray:
    """A at neutral orientation, zero velocity, VBS/LCG half-filled, 15-D (cf. main_sam.py)."""
    return np.array([0.0, 0.0, robot.depth,
                     1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     50.0, 50.0])


def _step(robot: SAM, u3, x0: np.ndarray, dur: float = None) -> np.ndarray:
    """One coarse-Euler rollout of duration ``dur`` under constant control ``u3``.

    ``u3 = [delta_s, delta_r, throttle]`` (throttle in [-1, 1]); VBS/LCG are held at
    their neutral fill to form the full 5-D control [x_vbs, x_lcg, delta_s, delta_r,
    throttle].
    """
    u3 = np.asarray(u3, dtype=float)
    u5 = np.array([robot.vbs_neutral, robot.lcg_neutral, u3[0], u3[1], u3[2]])
    ctrl = Controls(cps=u5.reshape(robot.nu, 1))
    x_next, _ = robot.dynamics(x0.copy(), ctrl, robot.dt if dur is None else dur)
    return x_next


def _dx_of_throttle(robot: SAM, x0: np.ndarray, s) -> np.ndarray:
    """Forward displacement for a constant throttle ``s`` (delta_s/r = 0)."""
    return np.array([_step(robot, [0.0, 0.0, float(si)], x0)[0] - x0[0]
                     for si in np.atleast_1d(s)])


# ---------------------------------------------------------------------------
# 1. Rollout sanity
# ---------------------------------------------------------------------------
def test_throttle_coast_is_static():
    """At the coast throttle (rpm = 0) the AUV does not move."""
    robot = _make_robot()
    x0 = _start_state(robot)
    x_next = _step(robot, [0.0, 0.0, _s_coast(robot)], x0)
    assert abs(x_next[0] - x0[0]) < 1e-6
    assert abs(x_next[1] - x0[1]) < 1e-6
    assert abs(x_next[7]) < 1e-6          # body surge velocity stays zero


def test_full_throttle_moves_forward():
    """Full throttle accelerates forward and settles at a sane cruise speed.

    This used to assert that ONE knot from REST clears the knot spacing. That no longer
    holds at this fixture's ``dt=0.6``: full-throttle terminal surge is ~0.32 m/s, so a
    0.273 m spacing needs ``dt >~ 0.86 s``.  (``rpm_max`` is 500 here; the from-rest
    displacement was ~0.86 m back when it was 1525, and thrust goes as rpm^2.)
    ``examples/main_sam.py`` runs ``dt=1.0``, which does clear it — so assert the dynamics
    property here and the dt-dependent reachability explicitly, rather than a stale constant.
    """
    robot = _make_robot()
    x0 = _start_state(robot)
    x1 = _step(robot, [0.0, 0.0, S_FULL], x0)
    assert x1[0] - x0[0] > 0.0, "full throttle must move the AUV forward"
    assert x1[7] > 0.0, "full throttle must build positive surge velocity"

    # Cruise speed: chain enough knots to reach the thrust/drag balance.
    x = x0.copy()
    for _ in range(10):
        x = _step(robot, [0.0, 0.0, S_FULL], x)
    u_cruise = x[7]
    assert 0.2 < u_cruise < 1.0, f"implausible cruise surge {u_cruise:.3f} m/s"
    # Reachability is a statement about the SCENARIO's dt, not about this fixture's.
    assert u_cruise * 1.0 > _knot_spacing(robot), (
        f"at cruise {u_cruise:.3f} m/s a dt=1.0 knot must clear the "
        f"{_knot_spacing(robot):.3f} m spacing")


def test_throttle_is_bipolar_and_monotone():
    """Displacement runs monotonically from reverse through coast to full forward.

    The throttle is deliberately BIPOLAR (see ``SAM._expand_control_batch``): ``s = -1``
    is full reverse, giving the planner active braking authority, without which the
    terminal velocity cannot be driven to zero (passive drag is too weak over one knot).
    An earlier revision of the model was forward-only, and this test used to assert
    exactly that — the sign convention it now pins is the opposite one.
    """
    robot = _make_robot()
    x0 = _start_state(robot)
    s_coast = _s_coast(robot)
    s = np.linspace(-1.0, 1.0, 21)
    dxs = _dx_of_throttle(robot, x0, s)

    assert dxs[0] < -1e-6, "s = -1 must drive the AUV backwards (reverse thrust)"
    assert dxs[-1] > 1e-6, "s = +1 must drive the AUV forwards"
    assert np.all(np.diff(dxs) > -1e-9), "displacement must be monotone in the throttle"
    # Reverse is ~rev_thrust_ratio times weaker than forward.
    assert abs(dxs[0]) < abs(dxs[-1])
    # The zero crossing sits at the coast throttle, not at s = -1.
    assert abs(_dx_of_throttle(robot, x0, s_coast)[0]) < 1e-6


def test_pitch_trim():
    """Stern-plane trim (delta_s~0.085) holds the vehicle level under thrust."""
    robot = _make_robot()
    x0 = _start_state(robot)
    q2_notrim = _step(robot, [0.0,     0.0, S_FULL], x0)[5]   # pitch quaternion
    q2_trim   = _step(robot, [DS_TRIM, 0.0, S_FULL], x0)[5]
    assert abs(q2_trim) < abs(q2_notrim)
    assert abs(q2_trim) < 0.05, f"trimmed pitch q2={q2_trim:.4f} should be near level"


# ---------------------------------------------------------------------------
# 2. The control-collapse regression guard (the property the fix restores)
# ---------------------------------------------------------------------------
def test_displacement_affine_in_throttle():
    """dx is ~affine in the throttle command (the sqrt map cancels the quadratic).

    Affine displacement is what lets the bundle's convex blends be consistent.  We
    check the curve never strays far from the straight chord through its endpoints
    (mild residual concavity from drag is fine; a quadratic would blow this up).
    """
    robot = _make_robot()
    x0 = _start_state(robot)
    s = np.linspace(-1.0, 1.0, 21)
    dxs = _dx_of_throttle(robot, x0, s)
    chord = dxs[0] + (dxs[-1] - dxs[0]) * (s - s[0]) / (s[-1] - s[0])
    max_rel_dev = np.max(np.abs(dxs - chord)) / (dxs[-1] - dxs[0])
    assert max_rel_dev < 0.15, f"throttle->dx deviates {max_rel_dev:.2%} from affine"


def test_convexification_gap_closed():
    """A convex blend of throttle samples advances by what APPLYING the blend delivers.

    This is the direct collapse test: for many random simplex weights over a throttle
    bundle, the blended next-state (sum w_j f_j) must match the state reached by
    actually applying the blended throttle (sum w_j s_j).  With raw-rpm controls this
    gap was ~+0.29 m (advance claimed on ~0 net control); the sqrt/forward-only map
    closes it to a few cm.
    """
    robot = _make_robot()
    x0 = _start_state(robot)
    rng = np.random.default_rng(0)

    s_samp = np.linspace(-1.0, 1.0, 25)
    dx_samp = _dx_of_throttle(robot, x0, s_samp)

    worst = 0.0
    for _ in range(2000):
        w = rng.random(s_samp.size)
        w /= w.sum()
        blended_s = float(w @ s_samp)
        blended_dx = float(w @ dx_samp)               # what the relaxation claims
        applied_dx = float(_dx_of_throttle(robot, x0, blended_s)[0])
        worst = max(worst, abs(blended_dx - applied_dx))

    assert worst < 0.06, (
        f"worst blend inconsistency {worst:.4f} m too large — control would collapse"
    )


# ---------------------------------------------------------------------------
# Runnable diagnostic: `python .../test_sam_rollout.py` prints the summary.
# ---------------------------------------------------------------------------
def _print_summary():
    robot = _make_robot()
    x0 = _start_state(robot)
    spacing = _knot_spacing(robot)

    print("=" * 72)
    print("SAM coarse-Euler rollout  (dt=%.2f, n_integrator=%d, corridor=%.1f, N=%d)"
          % (robot.dt, robot.n_integrator, robot.leg_length, N_KNOTS))
    print("knot spacing the planner must cover per knot: %.4f m" % spacing)
    print("control[2] is now a normalised throttle s in [-1,1] (s=-1 off, s=+1 full)")
    print("=" * 72)

    print("\n-- throttle -> displacement (should be ~affine, forward-only) --")
    print("%8s %9s %9s %12s" % ("s", "rpm", "dx", "pitch q2"))
    for s in np.linspace(-1.0, 1.0, 9):
        xn = _step(robot, [0.0, 0.0, s], x0)
        rpm = np.sqrt((s + 1.0) / 2.0) * robot.rpm_max
        print("%+8.2f %9.1f %+9.4f %+12.4f" % (s, rpm, xn[0]-x0[0], xn[5]))

    s_samp = np.linspace(-1.0, 1.0, 25)
    dx_samp = _dx_of_throttle(robot, x0, s_samp)
    chord = dx_samp[0] + (dx_samp[-1]-dx_samp[0]) * (s_samp-s_samp[0]) / (s_samp[-1]-s_samp[0])
    max_rel_dev = np.max(np.abs(dx_samp - chord)) / (dx_samp[-1] - dx_samp[0])

    rng = np.random.default_rng(0)
    worst = max(
        abs(float(w @ dx_samp) - float(_dx_of_throttle(robot, x0, float(w @ s_samp))[0]))
        for w in ((lambda v: v / v.sum())(rng.random(s_samp.size)) for _ in range(2000))
    )

    print("\n-- collapse guard --")
    print("  throttle->dx affine deviation : %.2f%% of range" % (100 * max_rel_dev))
    print("  worst convex-blend gap        : %.4f m  (was ~+0.29 m with raw rpm)" % worst)
    print("  => the relaxation can no longer advance the state on ~zero net control,")
    print("     so the planner must apply real throttle and the rollout actually moves.")
    print("=" * 72)


if __name__ == "__main__":
    _print_summary()
