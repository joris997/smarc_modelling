"""Per-sample quality filtering and segmentation.

This is the load-bearing part of the whole pipeline.  ``checkpoints/pinn.pt`` was
normalised over ``p in [-757.9, 1.9] rad/s`` — a mocap glitch — so every real roll rate
maps into the top 0.3% of the normalised range and **p, q, r are effectively dead inputs
in that model**.  The glitch lives in ``rosbag_5``, which ``notes.txt`` labels *"Good"*,
so label-based filtering alone cannot catch it: rejection has to be per-sample.

Five layers, applied in order:

===  =====================================================================
L0   structural   non-monotonic stamps, non-finite, bad |q|, out-of-range vbs_fb
L1   physical box hard bounds on nu -- the rule that un-poisons normalisation
L2   local        Hampel per channel + odom-twist vs differenced-mocap cross-check
L3   dilation     grow the invalid mask by the SG half-window
L4   segmentation split at drops and at recording gaps; keep long-enough runs
L5   bag verdict  drop bags with no usable segment or too many rejects
===  =====================================================================

Why fixed physical bounds and not quantiles: the 0.1% quantile of ``p`` over this corpus
is **-746.99 rad/s**.  ``rosbag_5`` alone contributes 14 glitch samples, so no quantile
clip at any sane level removes them.
"""
import dataclasses
from dataclasses import dataclass

import numpy as np

try:
    from . import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401


@dataclass(frozen=True)
class QualityConfig:
    """Every filtering knob.  Hashed into the cache key, so changing one rebuilds."""
    # L1 -- hard physical box on nu.  Legit measured maxima are |u| 0.64, |v| 0.32,
    # |w| 0.27 m/s; SAM cannot rotate at 2 rad/s in a test tank.
    v_max: float = 0.8                  # m/s   on u, v, w
    w_max: float = 2.0                  # rad/s on p, q, r
    # L0
    quat_tol: float = 1e-3
    vbs_lo: float = 0.0
    vbs_hi: float = 100.0
    # L2 -- a spike must be anomalous BOTH locally and on the channel's own scale.
    # The local MAD over a 7-sample window is ~10x smaller than the channel's global MAD
    # (measured: v 0.0029 vs 0.0160 m/s), so `nsigma * local_sigma` alone is an absurdly
    # tight absolute threshold that flags every genuine fast transient -- it rejected
    # 2.5% of clean samples, which after dilation cascaded to 82% of the corpus.
    # `hampel_floor_frac` puts a floor under it at a fraction of the GLOBAL channel MAD.
    hampel_window: int = 7
    hampel_nsigma: float = 6.0
    hampel_floor_frac: float = 1.0
    pos_xcheck_tol: float = 0.25        # m/s disagreement between twist and d(mocap)/dt
    # L3 -- the Savitzky-Golay window in targets.py is 9 (half-window 4), but dilating by
    # the full half-window fragments bags below `min_segment` and costs far more than the
    # residual leakage is worth (SG weights are already small at distance 3-4, and the
    # training losses are Huber).  +-2 is the measured knee.
    dilate: int = 2
    # L4
    gap_max: float = 0.25               # s; median dt is 0.1003
    #: >= the SG window (9) so every kept segment can produce a derivative.
    min_segment: int = 12
    # L5
    min_valid: int = 40
    max_reject_frac: float = 0.30

    def key(self):
        return tuple(sorted(dataclasses.asdict(self).items()))


def _quat_to_R(q):
    """(N,4) scalar-first unit quaternions -> (N,3,3) body->world rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def _hampel(x, window, nsigma, floor):
    """Rolling-median outlier mask for a 1-D series.  True == outlier.

    ``floor`` is an absolute lower bound on the flagging threshold, so a locally very
    smooth stretch (tiny local MAD) cannot make every ordinary transient an "outlier".
    """
    n = len(x)
    half = window // 2
    if n < window:
        return np.zeros(n, dtype=bool)
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n - 1)
    hi = np.clip(idx + half + 1, 1, n)
    med = np.empty(n)
    mad = np.empty(n)
    for i in range(n):
        w = x[lo[i]:hi[i]]
        med[i] = np.median(w)
        mad[i] = np.median(np.abs(w - med[i]))
    thresh = np.maximum(nsigma * 1.4826 * mad, max(floor, 1e-9))
    return np.abs(x - med) > thresh


def channel_scales(trajs, cfg):
    """Robust per-channel MAD-sigma of ``nu``, over in-box samples of every bag.

    Used both as the Hampel floor and (in ``targets.py``) as the rollout-loss whitening.
    Computed on the box-filtered corpus so the p=-758 glitch cannot set the scale.
    """
    allnu = np.concatenate([t.nu for t in trajs])
    box = ((np.abs(allnu[:, :3]) <= cfg.v_max).all(1)
           & (np.abs(allnu[:, 3:]) <= cfg.w_max).all(1))
    sub = allnu[box]
    return 1.4826 * np.median(np.abs(sub - np.median(sub, 0)), axis=0)


def _dilate(mask, k):
    """Grow a boolean mask by +-k samples (no scipy dependency)."""
    if k <= 0 or not mask.any():
        return mask
    out = mask.copy()
    for s in range(1, k + 1):
        out[s:] |= mask[:-s]
        out[:-s] |= mask[s:]
    return out


def apply_quality(traj, cfg=QualityConfig(), scales=None):
    """Fill ``traj.valid`` / ``traj.seg_id`` / ``traj.reject`` in place.  Returns ``traj``.

    ``scales`` is the per-channel MAD-sigma from :func:`channel_scales` (corpus-wide, so
    it must be passed in); ``None`` disables the Hampel floor.
    """
    n = len(traj)
    rej = {}

    # --- L0 structural ----------------------------------------------------
    # 67 of 120 bags have non-monotonic header stamps (172 samples with dt <= 0, down to
    # -0.147 s).  np.gradient(nu, time) divides by exactly those.
    nonmono = np.zeros(n, dtype=bool)
    running = -np.inf
    for i, ti in enumerate(traj.t):
        if not (ti > running):
            nonmono[i] = True
        else:
            running = ti
    rej["nonmonotonic_time"] = nonmono

    rej["nonfinite"] = ~(np.isfinite(traj.eta).all(1) & np.isfinite(traj.nu).all(1)
                         & np.isfinite(traj.u_fb).all(1) & np.isfinite(traj.t))
    rej["bad_quat"] = np.abs(np.linalg.norm(traj.eta[:, 3:], axis=1) - 1.0) > cfg.quat_tol
    # vbs_fb is a sensor read and undershoots its own range (measured min -2.0).
    rej["vbs_out_of_range"] = ((traj.u_fb[:, 0] < cfg.vbs_lo - 5.0)
                               | (traj.u_fb[:, 0] > cfg.vbs_hi + 5.0))
    traj.u_fb[:, 0] = np.clip(traj.u_fb[:, 0], cfg.vbs_lo, cfg.vbs_hi)

    # --- L1 hard physical box --------------------------------------------
    rej["speed_box"] = (np.abs(traj.nu[:, :3]) > cfg.v_max).any(1)
    rej["rate_box"] = (np.abs(traj.nu[:, 3:]) > cfg.w_max).any(1)

    base = np.zeros(n, dtype=bool)
    for m in rej.values():
        base |= m

    # --- L2 local consistency --------------------------------------------
    # Hampel is only meaningful on samples that survived the box; a +-758 rad/s spike
    # would otherwise dominate its own rolling median.
    hampel = np.zeros(n, dtype=bool)
    keep = ~base
    if keep.sum() >= cfg.hampel_window:
        sub = traj.nu[keep]
        flag = np.zeros(sub.shape[0], dtype=bool)
        for c in range(6):
            floor = 0.0 if scales is None else cfg.hampel_floor_frac * float(scales[c])
            flag |= _hampel(sub[:, c], cfg.hampel_window, cfg.hampel_nsigma, floor)
        hampel[np.flatnonzero(keep)] = flag
    rej["hampel"] = hampel

    # The odom twist and the differenced mocap position are two views of the same thing;
    # where they disagree, one of them glitched.
    xcheck = np.zeros(n, dtype=bool)
    dt = np.diff(traj.t)
    good_dt = dt > 1e-6
    if good_dt.any():
        R = _quat_to_R(traj.eta[:, 3:])
        v_world = np.zeros((n, 3))
        v_world[:-1][good_dt] = (np.diff(traj.eta[:, :3], axis=0)[good_dt]
                                 / dt[good_dt, None])
        v_body = np.einsum("nji,nj->ni", R, v_world)          # R^T @ v_world
        err = np.linalg.norm(v_body - traj.nu[:, :3], axis=1)
        cmp_ok = np.zeros(n, dtype=bool)
        cmp_ok[:-1] = good_dt
        xcheck = cmp_ok & (err > cfg.pos_xcheck_tol) & ~base
    rej["pos_xcheck"] = xcheck

    invalid = base | hampel | xcheck

    # --- L3 dilation ------------------------------------------------------
    dilated = _dilate(invalid, cfg.dilate)
    rej["dilation"] = dilated & ~invalid
    invalid = dilated

    # --- L4 segmentation --------------------------------------------------
    seg = np.full(n, -1, dtype=int)
    sid, cur = 0, []
    for i in range(n):
        if invalid[i]:
            if cur:
                sid += 1
            cur = []
            continue
        if cur and (traj.t[i] - traj.t[cur[-1]]) > cfg.gap_max:
            sid += 1
            cur = []
        cur.append(i)
        seg[i] = sid
    # Drop segments that are too short to derive or roll out through.
    for s in np.unique(seg[seg >= 0]):
        m = seg == s
        if m.sum() < cfg.min_segment:
            seg[m] = -1
            invalid |= m

    traj.valid = ~invalid
    traj.seg_id = seg
    traj.reject = rej
    return traj


def bag_verdict(traj, cfg=QualityConfig()):
    """``(usable, why)`` -- L5.  ``why`` is "" when usable."""
    if traj.label.quality == "DNU":
        return False, "DNU label"
    if traj.n_valid < cfg.min_valid:
        return False, f"only {traj.n_valid} usable samples (< {cfg.min_valid})"
    hard = (traj.reject.get("speed_box", 0) | traj.reject.get("rate_box", 0)
            | traj.reject.get("hampel", 0) | traj.reject.get("pos_xcheck", 0))
    frac = float(np.sum(hard)) / max(len(traj), 1)
    if frac > cfg.max_reject_frac:
        return False, f"{frac:.0%} of samples failed L1/L2 (> {cfg.max_reject_frac:.0%})"
    return True, ""


def print_report(trajs, cfg=QualityConfig(), top=20):
    """Corpus-level summary + the worst offenders."""
    import collections
    rule_tot = collections.Counter()
    raw = val = 0
    per_bag = []
    for tr in trajs:
        raw += len(tr)
        val += tr.n_valid
        for k, m in tr.reject.items():
            rule_tot[k] += int(np.sum(m))
        per_bag.append((len(tr) - tr.n_valid, tr))

    print(f"\n=== quality report ({len(trajs)} bags) ===")
    print(f"  samples: {raw} raw -> {val} valid ({100.0 * val / max(raw, 1):.1f}%)")
    print("  rejections by rule:")
    for k, v in rule_tot.most_common():
        print(f"    {k:22s} {v:6d}")

    by_q = {}
    for tr in trajs:
        q = tr.label.quality
        a, b, c = by_q.get(q, (0, 0, 0))
        by_q[q] = (a + 1, b + len(tr), c + tr.n_valid)
    print("  by notes.txt label:      bags   raw  valid")
    for q, (nb, nr, nv) in sorted(by_q.items()):
        print(f"    {q:12s} {nb:6d} {nr:6d} {nv:6d}")

    print(f"  worst {top} bags by dropped samples:")
    for dropped, tr in sorted(per_bag, key=lambda x: -x[0])[:top]:
        ok, why = bag_verdict(tr, cfg)
        rates = np.abs(tr.nu[:, 3:]).max() if len(tr) else 0.0
        print(f"    {tr.name:14s} {tr.label.quality:4s} dropped {dropped:4d}/{len(tr):4d}"
              f"  max|w| {rates:9.2f}  {'' if ok else 'UNUSABLE: ' + why}")
