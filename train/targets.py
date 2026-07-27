"""Regression targets, derivatives and normalisation statistics.

The one-step target reproduces ``piml/utils/utility_functions.py::load_data_from_bag``:

    v_dot_nod = Minv (tau - C nu - g)        # acceleration if there were no damping
    Dv_target = M (v_dot_nod - nu_dot)       # == tau - C nu - g - M nu_dot

so a damping model is asked for ``D(x) nu ~ Dv_target``.  Two departures from that
function, both switchable so the port can be shown faithful first:

* **derivative** -- ``np.gradient`` on 10 Hz mocap velocity is very noisy AND divides by
  the 172 non-positive ``dt`` this corpus contains.  We use a local quadratic fit on the
  ACTUAL timestamps (a Savitzky-Golay generalisation to non-uniform grids), which needs
  no resampling and is zero-phase.  Measured on rosbag_3, std of ``nu_dot``:
  u 0.058->0.031, v 0.048->0.010, w 0.089->0.014, q 0.225->0.036, r 0.100->0.020.
* **quaternion order** -- see ``cdr.read_bag_raw``.

Read the honest caveat before trusting this target: over the Good bags, ordinary least
squares gives R^2 = 0.066 for the best CONSTANT FULL 6x6 D, while a constant bias with no
D at all gives 0.140, and the LS-optimal symmetric D is *indefinite*.  The residual is
dominated by white-box thrust (tau rms 23.5 in surge) and buoyancy (g rms 5.6 in pitch)
error, which no positive-definite D can represent.  That is why this is only a Stage-A
*pretraining* signal and the real objective is the multi-step rollout in ``rollout.py``.
``fit_bias`` quantifies the part D cannot reach.
"""
import numpy as np

try:
    from . import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from smarc_modelling.piml.pinn.damping import CLAMP_LO, CLAMP_HI

#: Local-fit window (samples) and polynomial order for the derivative.
DERIV_WINDOW = 9
DERIV_ORDER = 2


def local_poly_derivative(t, y, window=DERIV_WINDOW, order=DERIV_ORDER):
    """d/dt of ``y`` (N,) or (N,C) by a local least-squares polynomial fit.

    Savitzky-Golay generalised to a NON-UNIFORM grid: for each sample, fit
    ``y ~ sum_k c_k (t - t_i)^k`` over the surrounding window and take ``c_1``.  On a
    uniform grid this is exactly ``savgol_filter(..., deriv=1)``; on this corpus's
    jittered stamps it is what ``savgol`` would need a resampling step to approximate.
    """
    y = np.asarray(y, dtype=float)
    squeeze = y.ndim == 1
    if squeeze:
        y = y[:, None]
    n, c = y.shape
    out = np.zeros_like(y)
    if n < order + 1:
        return out[:, 0] if squeeze else out
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        # Keep the stencil `window` wide near the ends rather than shrinking it, so the
        # fit stays well conditioned (a 3-point quadratic fit interpolates the noise).
        if hi - lo < order + 2:
            lo, hi = max(0, min(lo, n - (order + 2))), min(n, max(hi, order + 2))
        dt = t[lo:hi] - t[i]
        V = np.vander(dt, order + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(V, y[lo:hi], rcond=None)
        out[i] = coef[1]
    return out[:, 0] if squeeze else out


def gradient_derivative(t, y):
    """The legacy ``np.gradient`` derivative, for the faithfulness check only."""
    y = np.asarray(y, dtype=float)
    return np.stack([np.gradient(y[:, c], t) for c in range(y.shape[1])], axis=1)


def feature_vector(traj):
    """(N,12) network input ``[nu(6), vbs, lcg, dS, dR, rpm1, rpm2]``.

    This must match what ``SAM_torch._dyn`` feeds at inference EXACTLY, or training and
    deployment see different distributions.  That path uses the *command* with VBS/LCG
    clamped to [0,100] (``uv``/``ul`` at SAM_torch.py:507-508), not the actuator state --
    so ``u_cmd`` here, not ``u_fb``.
    """
    u = traj.u_cmd.copy()
    u[:, 0] = np.clip(u[:, 0], 0.0, 100.0)
    u[:, 1] = np.clip(u[:, 1], 0.0, 100.0)
    return np.concatenate([traj.nu, u], axis=1)


def compute_bag_targets(traj, sam=None, deriv="localpoly", quiet=True):
    """Per-sample white-box decomposition and damping target for one bag.

    Returns a dict of (N, ...) arrays.  Rows outside a valid segment are computed but
    should be masked with ``traj.valid``; the derivative is evaluated PER SEGMENT so it
    never differentiates across a dropped sample or a recording gap.
    """
    from smarc_modelling.vehicles.SAM import SAM
    if sam is None:
        sam = SAM(0.02)

    n = len(traj)
    nu_dot = np.zeros((n, 6))
    fn = local_poly_derivative if deriv == "localpoly" else gradient_derivative
    for sl in traj.segments(min_len=DERIV_ORDER + 2):
        nu_dot[sl] = fn(traj.t[sl], traj.nu[sl])

    state19 = traj.state19()
    u_cmd = traj.u_cmd
    dt = np.diff(traj.t)
    dt_mean = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.1

    M = np.zeros((n, 6, 6))
    Cnu = np.zeros((n, 6))
    g_vec = np.zeros((n, 6))
    tau = np.zeros((n, 6))
    for i in range(n):
        sam.update_dt(float(dt[i]) if i < n - 1 and dt[i] > 0 else dt_mean)
        sam.dynamics(state19[i], u_cmd[i])     # populates .M .C .g_vec .tau in place
        M[i] = sam.M
        Cnu[i] = sam.C @ traj.nu[i]
        g_vec[i] = sam.g_vec
        tau[i] = sam.tau

    Mnu_dot = np.einsum("nij,nj->ni", M, nu_dot)
    y = tau - Cnu - g_vec - Mnu_dot            # == M (v_dot_nod - nu_dot)
    return {"feat": feature_vector(traj), "nu": traj.nu, "nu_dot": nu_dot, "y": y,
            "M": M, "Cnu": Cnu, "g": g_vec, "tau": tau, "Mnu_dot": Mnu_dot,
            "valid": traj.valid}


def robust_stats(feat, floor_frac=0.05):
    """``(x_mu, x_sigma)`` -- median and MAD-sigma, the standardisation the model stores.

    Deliberately NOT min/max: ``pinn.pt`` used a min/max box for both clamping and
    scaling, so the single ``p = -757.9 rad/s`` glitch set ``x_range[3] = 759.8`` and
    every real roll rate normalised into the top 0.3% of [0,1] -- p, q and r are dead
    inputs in that model.  Median/MAD over the (already box-filtered) training split
    cannot be moved by an outlier at all.

    The floor is a fraction of each channel's CLAMP-BOX half-width, not an absolute
    constant.  Some channels are genuinely degenerate in this corpus -- ``lcg_cmd`` is
    exactly 75.0 for every training sample, so its MAD is 0 -- and an absolute 1e-3 floor
    would amplify any deviation from 75 by 50,000x, handing the planner (which does move
    the LCG) an input several thousand sigma out.  Scaling to the box keeps a degenerate
    channel merely uninformative instead of explosive.
    """
    mu = np.median(feat, axis=0)
    sigma = 1.4826 * np.median(np.abs(feat - mu), axis=0)
    floor = floor_frac * 0.5 * (np.asarray(CLAMP_HI) - np.asarray(CLAMP_LO))
    return mu, np.maximum(sigma, floor)


def fit_bias(y, w=None):
    """Per-DOF median of the target -- the constant offset no PD ``D nu`` can produce.

    Reported, never shipped.  It is the visible part of the white-box thrust/buoyancy
    error and sets the ceiling on what Stage A can achieve.
    """
    return np.median(y, axis=0) if w is None else np.median(y, axis=0)


def whiten_scales(y, bias=None, floor=1e-6):
    """MAD-sigma of the target, per DOF.

    Without this the loss is entirely surge: measured target rms is
    ``[23.5, 2.2, 2.3, 0.26, 4.06, 3.3]``, so an unweighted L2 fits u and ignores p.
    """
    b = np.zeros(y.shape[1]) if bias is None else bias
    return np.maximum(1.4826 * np.median(np.abs(y - b), axis=0), floor)


def clamp_box():
    """The physical clamp box, from the runtime model so there is one source of truth."""
    return np.array(CLAMP_LO), np.array(CLAMP_HI)
