"""SAM AUV turning diagnostics — *which conditions inhibit a yaw turn?*

Motivation
----------
Driven by the Graphs-of-Bundles planner, SAM cannot start at ``(0,0,0)`` and reach a
goal that needs a slight yaw turn (e.g. ``(14,1,0)``), and it fails for BOTH damping
models (Fossen constant ``D`` and the learned PINN ``D``).  That the damping model does
not matter points away from drag and toward either (a) a control-regime problem (the
rudder only has authority once there is thrust to vector) or (b) a wiring problem in the
vehicle dynamics (the propeller-moment block is computed by a cross product and then has
its components index-swapped — see ``calculate_propeller_force`` / ``_propeller_force``).

This file is the diagnostic that tells those apart.  It sweeps the candidate conditions
one at a time and, for each, measures the response on the YAW axis versus the ROLL axis:

  * ``r_dot``  = instantaneous body yaw angular acceleration  (dX[12], at rest)
  * ``p_dot``  = instantaneous body roll angular acceleration (dX[10], at rest)
  * ``dpsi``   = world heading change after one ``dt`` rollout
  * ``droll``  = world roll-angle change after one ``dt`` rollout
  * ``dy``     = world lateral displacement after one ``dt`` rollout

The instantaneous accelerations (taken at rest, nu=0, where the only moment is the
commanded one) are the cleanest axis probe: a rudder command SHOULD show up as ``r_dot``
(yaw); if it shows up as ``p_dot`` (roll) instead, the moment is wired to the wrong axis.

Run as a script for the readable report::

    python classes/robots/smarc_modelling/test/test_sam_turn.py

or under pytest (needs the ``admm`` conda env; no GCS solve / Mosek required)::

    pytest classes/robots/smarc_modelling/test/test_sam_turn.py -v

The pytest assertions encode the *expected-correct* behaviour (rudder -> yaw, not roll),
so they document the defect by failing against broken dynamics and pass once it is fixed.
"""
import pathlib
import sys

import numpy as np

#   .../bundle-stl/classes/robots/smarc_modelling/test/test_sam_turn.py
#   parents[0]=test [1]=smarc_modelling [2]=robots [3]=utils [4]=bundle-stl
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SMARC_SRC = REPO_ROOT / "utils" / "robots" / "smarc_modelling" / "src"
if str(_SMARC_SRC) not in sys.path:
    sys.path.insert(0, str(_SMARC_SRC))

from utils.robots.sam import SAM            # noqa: E402
from utils.controls import Controls           # noqa: E402
from smarc_modelling.vehicles.SAM_torch import SAMTorch  # noqa: E402

# State-vector indices (15-D SAM state: eta(7) | nu(6) | act(2)=[vbs, lcg]).
QUAT = slice(3, 7)      # q0,q1,q2,q3 (q0 scalar)
P_IDX = 10              # body roll rate  p
Q_IDX = 11              # body pitch rate q
R_IDX = 12              # body yaw rate   r


# ---------------------------------------------------------------------------
# Setup helpers (mirror test_sam_rollout.py)
# ---------------------------------------------------------------------------
def _make_robot(piml_type=None) -> SAM:
    """A SAM with the same geometry as examples/main_sam.py (Fossen D by default)."""
    return SAM(dt=1.0, n_integrator=50, leg_length=15.0, piml_type=piml_type)


def _rest_state(robot: SAM) -> np.ndarray:
    """Origin, identity quaternion, zero velocity, VBS/LCG half-filled (15-D)."""
    return np.array([0.0, 0.0, robot.depth,
                     1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.5, 0.5])     # NORMALISED (SAM.ACT_SCALE)


def _s_coast(robot: SAM) -> float:
    """The throttle that produces rpm = 0.

    The throttle is BIPOLAR: ``s = -1`` is full REVERSE, not "off".  The coast point is
    where the signed thrust proxy ``P(s)`` crosses zero (see ``SAM._expand_control_batch``
    and the same helper in ``test_sam_rollout.py``); it is ≈ -0.818, not -1.
    """
    P_fwd = robot.rpm_max ** 2
    P_rev = -(robot.rpm_rev_max ** 2) / robot.rev_thrust_ratio
    return 2.0 * (-P_rev) / (P_fwd - P_rev) - 1.0


def _u5(robot: SAM, delta_s=0.0, delta_r=0.0, throttle=None, vbs=None, lcg=None):
    """Full 5-D reduced control [x_vbs, x_lcg, delta_s, delta_r, throttle].

    ``throttle=None`` means COAST (rpm 0), which is what these tests want when they say
    "off" — it used to be spelled ``-1.0``, but that is full reverse under the bipolar map.
    """
    vbs = robot.vbs_neutral if vbs is None else vbs
    lcg = robot.lcg_neutral if lcg is None else lcg
    throttle = _s_coast(robot) if throttle is None else throttle
    return np.array([vbs, lcg, delta_s, delta_r, throttle], dtype=float)


def _step(robot: SAM, u5, x0: np.ndarray) -> np.ndarray:
    """One coarse-Euler rollout of one ``dt`` under constant control ``u5``."""
    ctrl = Controls(cps=np.asarray(u5, float).reshape(robot.nu, 1))
    x_next, _ = robot.dynamics(x0.copy(), ctrl, robot.dt)
    return x_next


def _accel(robot: SAM, u5, x0: np.ndarray) -> np.ndarray:
    """Instantaneous dX = f(x0, u) via the shared torch model (expand 5-D -> 6-D u_ref).

    Evaluated at ``x0`` (typically rest, nu=0) so the body-rate derivatives dX[10:13]
    isolate the commanded moment on each axis without integration mixing them.
    """
    u_ref = robot._expand_control_batch(np.asarray(u5, float)[None, :])   # (1,6)
    # SAMTorch works in the vehicle's units, so the planner state's normalised VBS/LCG
    # must be converted before bypassing the wrapper's rollout (see SAM.ACT_SCALE).
    dX = robot._sam_torch.dynamics(robot.to_physical(x0[None, :]), u_ref)  # (1,15)
    return dX[0]


def _euler_angles(q: np.ndarray):
    """(roll, pitch, yaw) in rad from a quaternion [q0(scalar), q1, q2, q3]."""
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _turn_metrics(robot: SAM, u5, x0: np.ndarray) -> dict:
    """All yaw-vs-roll signals for one control, at rest x0."""
    dX = _accel(robot, u5, x0)
    xn = _step(robot, u5, x0)
    r0 = _euler_angles(x0[QUAT])
    rn = _euler_angles(xn[QUAT])
    return {
        "r_dot": dX[R_IDX], "p_dot": dX[P_IDX], "q_dot": dX[Q_IDX],
        "dpsi": rn[2] - r0[2], "droll": rn[0] - r0[0], "dpitch": rn[1] - r0[1],
        "dy": xn[1] - x0[1], "r": xn[R_IDX], "p": xn[P_IDX],
    }


# Throttle -> rpm for labelling the report (matches sam._expand_control_batch).
def _rpm(robot: SAM, s: float) -> float:
    return float(np.sqrt((np.clip(s, -1.0, 1.0) + 1.0) / 2.0) * robot.rpm_max)


THROTTLES = [-1.0, -0.6, -0.3, 0.0, 0.5, 1.0]
RUDDERS = [0.0, 0.05, 0.1, 0.3]


# ---------------------------------------------------------------------------
# 1. Rudder authority vs thrust — the central diagnostic
# ---------------------------------------------------------------------------
def test_rudder_yaws_not_rolls_at_thrust():
    """At meaningful thrust, a rudder command must accelerate YAW, not ROLL.

    The clean axis probe: at rest the only moment is the commanded one, so a positive
    delta_r should give |r_dot| (yaw) >> |p_dot| (roll).  If the propeller moment is
    wired to the wrong axis the inequality flips — this assertion catches that.
    """
    robot = _make_robot()
    x0 = _rest_state(robot)
    m = _turn_metrics(robot, _u5(robot, delta_r=0.1, throttle=1.0), x0)
    assert abs(m["r_dot"]) > 1e-3, f"rudder produced negligible yaw accel r_dot={m['r_dot']:.2e}"
    assert abs(m["r_dot"]) > abs(m["p_dot"]), (
        f"rudder authority is on the WRONG axis: |r_dot(yaw)|={abs(m['r_dot']):.3e} "
        f"should exceed |p_dot(roll)|={abs(m['p_dot']):.3e}")


def test_rudder_authority_grows_with_thrust():
    """Yaw authority should be ~zero at rpm 0 and grow with throttle (thrust-vectoring)."""
    robot = _make_robot()
    x0 = _rest_state(robot)
    yaw_off = abs(_turn_metrics(robot, _u5(robot, delta_r=0.1), x0)["r_dot"])
    yaw_full = abs(_turn_metrics(robot, _u5(robot, delta_r=0.1, throttle=1.0), x0)["r_dot"])
    assert yaw_off < 1e-4, f"rudder should be inert at rpm 0, got r_dot={yaw_off:.2e}"
    assert yaw_full > yaw_off, "yaw authority should increase with thrust"


def test_rudder_sign_controls_turn_direction():
    """Opposite rudder deflections produce opposite-sign yaw accelerations."""
    robot = _make_robot()
    x0 = _rest_state(robot)
    rp = _turn_metrics(robot, _u5(robot, delta_r=+0.1, throttle=1.0), x0)["r_dot"]
    rm = _turn_metrics(robot, _u5(robot, delta_r=-0.1, throttle=1.0), x0)["r_dot"]
    assert np.sign(rp) == -np.sign(rm) and abs(rp) > 1e-3


# ---------------------------------------------------------------------------
# 2. Stern plane drives pitch, not yaw
# ---------------------------------------------------------------------------
def test_stern_plane_pitches_not_yaws():
    robot = _make_robot()
    x0 = _rest_state(robot)
    m = _turn_metrics(robot, _u5(robot, delta_s=0.1, throttle=1.0), x0)
    assert abs(m["q_dot"]) > abs(m["r_dot"]), "delta_s should drive pitch, not yaw"


# ---------------------------------------------------------------------------
# 3. Quaternion handling: heading preserved & kinematically consistent
# ---------------------------------------------------------------------------
def _yaw_quat(psi: float) -> np.ndarray:
    return np.array([np.cos(psi / 2.0), 0.0, 0.0, np.sin(psi / 2.0)])


def test_quaternion_heading_preserved_under_zero_control():
    """A pre-yawed vehicle, coasting at rpm 0, keeps its heading (no spurious drift)."""
    robot = _make_robot()
    x0 = _rest_state(robot)
    x0[QUAT] = _yaw_quat(0.2)
    xn = _step(robot, _u5(robot), x0)
    assert abs(_euler_angles(xn[QUAT])[2] - 0.2) < 1e-3


def test_yaw_kinematics_consistent():
    """A known body yaw rate r integrates to dpsi ~ r*dt (quaternion kinematics sane).

    Probed over a SHORT horizon so rotational damping (which would otherwise bleed the
    rate off over a full dt) is negligible and the kinematic relation is clean.
    """
    robot = _make_robot()
    short = SAM(dt=0.05, n_integrator=5, leg_length=15.0)
    short._sam_torch = robot._sam_torch        # reuse the model; only dt differs
    x0 = _rest_state(short)
    x0[R_IDX] = 0.05                           # small yaw rate, zero control
    xn = _step(short, _u5(short), x0)
    assert abs((_euler_angles(xn[QUAT])[2]) - 0.05 * short.dt) < 5e-4


# ---------------------------------------------------------------------------
# 4. Torch <-> scalar parity on the yaw axis (so any fix lands in both paths)
# ---------------------------------------------------------------------------
def test_yaw_axis_torch_scalar_parity():
    """The wrapper rollout and a direct SAMTorch derivative agree on the yaw axis."""
    robot = _make_robot()
    x0 = _rest_state(robot)
    u5 = _u5(robot, delta_r=0.1, throttle=1.0)
    # Direct derivative and a 1-substep wrapper step should agree to high precision on r.
    rdot = _accel(robot, u5, x0)[R_IDX]
    one_step = SAM(dt=robot.dt, n_integrator=1, leg_length=15.0)
    one_step._sam_torch = robot._sam_torch
    xn = _step(one_step, u5, x0)
    assert abs(xn[R_IDX] - rdot * robot.dt) < 1e-6


# ---------------------------------------------------------------------------
# Runnable diagnostic report
# ---------------------------------------------------------------------------
def _print_sweep(robot, x0, title):
    print("\n" + title)
    print("%8s %8s %8s | %10s %10s | %10s %10s %9s"
          % ("throttle", "rpm", "delta_r", "r_dot", "p_dot", "dpsi", "droll", "dy"))
    for s in THROTTLES:
        for dr in RUDDERS:
            m = _turn_metrics(robot, _u5(robot, delta_r=dr, throttle=s), x0)
            print("%8.2f %8.0f %8.3f | %10.3e %10.3e | %10.3e %10.3e %9.4f"
                  % (s, _rpm(robot, s), dr, m["r_dot"], m["p_dot"],
                     m["dpsi"], m["droll"], m["dy"]))


def _print_summary():
    print("=" * 96)
    print("SAM turn diagnostics  (dt=1.0, n_integrator=50, corridor=15.0)")
    print("Axis probe at rest: r_dot = yaw accel (dX[12]), p_dot = roll accel (dX[10]).")
    print("A working rudder => |r_dot| >> |p_dot| once thrust is up.")
    print("=" * 96)

    for tag, piml in [("Fossen (constant D)", None), ("PINN (learned D)", "pinn")]:
        try:
            robot = _make_robot(piml_type=piml)
        except Exception as e:                       # PINN checkpoint may be absent
            print("\n[skip] %s: %s" % (tag, e))
            continue
        x0 = _rest_state(robot)
        _print_sweep(robot, x0, "-- rudder x throttle sweep :: %s --" % tag)

    robot = _make_robot()
    x0 = _rest_state(robot)
    print("\n-- stern plane (delta_s) at full throttle: should be pitch, not yaw --")
    for ds in [0.0, 0.05, 0.1]:
        m = _turn_metrics(robot, _u5(robot, delta_s=ds, throttle=1.0), x0)
        print("  delta_s=%5.2f  q_dot(pitch)=%+.3e  r_dot(yaw)=%+.3e" % (ds, m["q_dot"], m["r_dot"]))

    print("\n-- VBS / LCG at full throttle: should be depth/pitch, not yaw --")
    for vbs in [0.4, 0.5, 0.6]:      # NORMALISED (SAM.ACT_SCALE)
        m = _turn_metrics(robot, _u5(robot, vbs=vbs, throttle=1.0), x0)
        print("  vbs=%5.2f  r_dot(yaw)=%+.3e" % (vbs, m["r_dot"]))

    print("\n-- quaternion handling --")
    xq = _rest_state(robot); xq[QUAT] = _yaw_quat(0.2)
    yaw_keep = _euler_angles(_step(robot, _u5(robot), xq)[QUAT])[2]
    xr = _rest_state(robot); xr[R_IDX] = 0.05
    dpsi_kin = _euler_angles(_step(robot, _u5(robot), xr)[QUAT])[2]
    print("  pre-yawed 0.200 rad, zero ctrl -> heading %.4f (drift %.1e)"
          % (yaw_keep, yaw_keep - 0.2))
    print("  body r=0.05, dt=1.0 -> dpsi %.4f (expect ~%.4f)" % (dpsi_kin, 0.05 * robot.dt))
    print("=" * 96)


if __name__ == "__main__":
    _print_summary()
