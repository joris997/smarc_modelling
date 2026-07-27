"""Differentiable multi-step rollout and the loss terms.

This is the objective that actually matters.  The one-step algebraic target
(``targets.py``) is dominated by white-box thrust/buoyancy error -- least squares reaches
R^2 = -0.071 for the best constant full ``D`` -- so fitting a PD ``D`` to it mostly teaches
the model to blow up near ``nu = 0``.  Rolling the real dynamics forward and matching the
recorded velocity avoids that: it never differentiates the noisy 10 Hz mocap, and it scores
the model on the quantity the benchmark reports.

Two ``SAMTorch`` details this relies on:

* ``rollout()`` and ``dynamics()`` wrap in ``torch.no_grad()``, so we drive ``_dyn`` /
  ``_step_rk4`` directly instead.
* ``piml_model=`` (ctor kwarg) injects the live, gradient-carrying model, so no checkpoint
  needs to exist before training starts.
"""
import numpy as np
import torch

try:
    from . import _bootstrap, targets  # noqa: F401
except ImportError:
    import _bootstrap, targets  # noqa: F401

from smarc_modelling.piml.pinn.damping import D_WHITEBOX
from smarc_modelling.vehicles.SAM_torch import SAMTorch


class WindowBank:
    """All training samples flattened, plus the bookkeeping to cut rollout windows.

    Windows never cross a segment boundary, so they never span a dropped sample or a
    recording gap -- ``seg`` carries a globally unique id per (bag, segment).
    """

    def __init__(self, trajs, names, weights, device, dtype=torch.float32,
                 deriv="localpoly", sam=None):
        X, U, T, NU, FEAT, Y, M, SEG, W, BAG = [], [], [], [], [], [], [], [], [], []
        seg_offset = 0
        for bi, name in enumerate(names):
            tr = trajs[name]
            d = targets.compute_bag_targets(tr, sam=sam, deriv=deriv)
            keep = tr.valid
            if not keep.any():
                continue
            seg = tr.seg_id.copy()
            seg[~keep] = -1
            seg[keep] += seg_offset
            seg_offset = seg[keep].max() + 1

            X.append(tr.state15()); U.append(tr.u_cmd); T.append(tr.t); NU.append(tr.nu)
            FEAT.append(d["feat"]); Y.append(d["y"]); M.append(d["M"])
            SEG.append(seg)
            W.append(np.full(len(tr), weights.get(name, 1.0)))
            BAG.append(np.full(len(tr), bi))

        def t_(a, dt=dtype):
            return torch.as_tensor(np.concatenate(a), dtype=dt, device=device)

        self.X = t_(X); self.U = t_(U); self.T = t_(T, torch.float64)
        self.NU = t_(NU); self.feat = t_(FEAT); self.y = t_(Y); self.M = t_(M)
        self.w = t_(W)
        self.seg = torch.as_tensor(np.concatenate(SEG), dtype=torch.long, device=device)
        self.bag = torch.as_tensor(np.concatenate(BAG), dtype=torch.long, device=device)
        self.valid = self.seg >= 0
        self.device, self.dtype = device, dtype
        self._starts = {}

    def __len__(self):
        return int(self.valid.sum())

    def starts(self, horizon):
        """Indices ``i`` where ``i .. i+horizon`` all lie in one segment."""
        if horizon in self._starts:
            return self._starts[horizon]
        seg = self.seg
        n = seg.numel()
        ok = torch.ones(n - horizon, dtype=torch.bool, device=seg.device)
        base = seg[: n - horizon]
        ok &= base >= 0
        for j in range(1, horizon + 1):
            ok &= seg[j: n - horizon + j] == base
        idx = torch.nonzero(ok, as_tuple=False).squeeze(1)
        self._starts[horizon] = idx
        return idx

    def onestep(self):
        """``(feat, nu, y, M, w)`` for every valid sample -- the Stage-A tensors."""
        v = self.valid
        return self.feat[v], self.NU[v], self.y[v], self.M[v], self.w[v]


def make_sim(model, device, dtype=torch.float32, tau_act=None):
    """A ``SAMTorch`` whose damping IS the model under training.

    ``tau_act`` is pinned rather than left to default to ``dt``: the default would make
    the actuator time constant follow the integration substep, so the physics would change
    with the curriculum horizon.  0.3 matches the planner's nominal knot horizon.
    """
    sim = SAMTorch(dt=0.02, device=str(device), dtype=dtype, piml_type="pinn",
                   piml_model=model, tau_act=0.3 if tau_act is None else tau_act,
                   compile_mode=None)
    return sim


def rollout_windows(sim, bank, idx, horizon, n_sub=2, integrator="rk4"):
    """Integrate ``horizon`` data intervals from each window start.

    Returns ``nu_hat (B, horizon, 6)`` -- the predicted body velocity at each recorded
    sample time.  Gradients flow through every substep.
    """
    step = sim._step_rk4 if integrator == "rk4" else sim._step_euler
    X = sim._normalize_quat(bank.X[idx])
    keep = torch.ones(X.shape[0], 1, dtype=torch.bool, device=X.device)
    out = []
    for j in range(horizon):
        u = bank.U[idx + j]
        dt = (bank.T[idx + j + 1] - bank.T[idx + j]).to(X.dtype).unsqueeze(1)
        h = dt / n_sub
        for _ in range(n_sub):
            X = step(X, u, h, keep)
        out.append(X[:, 7:13])
    return torch.stack(out, dim=1)


# ----------------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------------
def huber(e, delta):
    a = e.abs()
    return torch.where(a <= delta, 0.5 * e * e, delta * (a - 0.5 * delta))


def rollout_loss(nu_hat, nu_tgt, w_nu, weights, gamma=0.9, delta=2.0):
    """Discounted whitened Huber on the velocity trace.

    ``w_nu`` (the reciprocal per-channel MAD-sigma) is not optional: the natural scale of
    ``v`` is ~14x smaller than ``u``, so an unweighted loss fits surge and ignores
    everything else.
    """
    H = nu_hat.shape[1]
    disc = gamma ** torch.arange(H, device=nu_hat.device, dtype=nu_hat.dtype)
    e = (nu_hat - nu_tgt) * w_nu
    per = huber(e, delta).mean(-1) * disc                       # (B, H)
    return (per.mean(1) * weights).sum() / weights.sum().clamp_min(1e-9)


def data_loss(model, feat, nu, y, weights, w_y, bias, delta=2.0, min_speed=0.02):
    """Whitened Huber on ``D(x) nu`` against the debiased one-step target."""
    pred = model.damping_force(feat, nu)
    e = (pred - (y - bias)) * w_y
    m = (torch.linalg.vector_norm(nu, dim=1) > min_speed).to(e.dtype)
    per = huber(e, delta).mean(1) * m * weights
    return per.sum() / (m * weights).sum().clamp_min(1e-9)


def anchor_loss(model, feat):
    """Keep ``D`` near the white box where the data says nothing.

    Load-bearing rather than cosmetic: this tank corpus barely excites 5 of 6 DOFs (the
    MAD-sigma of ``v`` is 0.015 m/s), while the planner queries the whole clamp box.  This
    is the physics prior that fills in everywhere the data is silent.
    """
    D = model(feat)
    S = torch.as_tensor(np.diag(D_WHITEBOX).copy(), dtype=D.dtype, device=D.device)
    ref = torch.as_tensor(D_WHITEBOX, dtype=D.dtype, device=D.device)
    return (((D - ref) / S).pow(2).sum(dim=(1, 2))).mean()


def stiffness_loss(model, feat, Minv_sqrt, target=60.0):
    """Hinge on ``lambda_max(M^-1 D)``, the eigenvalue that sets the Euler step limit.

    ``M^-1 D`` is not symmetric but is similar to ``M^-1/2 D M^-1/2``, which is -- so
    ``eigvalsh`` on the symmetrised form gives the same spectrum at a fraction of the cost.
    Largely redundant with the model's hard ``lambda_cap``; it shapes the interior rather
    than the bound.
    """
    D = model(feat)
    S = Minv_sqrt @ D @ Minv_sqrt
    lam = torch.linalg.eigvalsh(0.5 * (S + S.transpose(-2, -1)))[:, -1]
    return torch.relu(lam - target).pow(2).mean()


def subsample(n, k, rng, device):
    """Row indices for a stochastic estimate of the full-batch regularisers.

    The anchor and stiffness terms are averages over the training set, and evaluating them
    on all ~4.4k rows every minibatch dominated the step time (the stiffness term runs a
    batched ``eigvalsh``).  A fresh random subset per step is an unbiased estimate at a
    fraction of the cost.
    """
    if k >= n:
        return slice(None)
    return torch.as_tensor(rng.integers(0, n, size=k), device=device)


def inv_sqrt_spd(M):
    """``M^-1/2`` for a symmetric positive-definite matrix."""
    ev, V = np.linalg.eigh(M)
    return V @ np.diag(ev ** -0.5) @ V.T


@torch.no_grad()
def eval_rollout(sim, bank, horizon, w_nu, n_sub=2, integrator="rk4", max_windows=4096,
                 chunk=1024):
    """Un-discounted whitened RMSE over every window of the given horizon.

    This is the model-selection metric: it is the same quantity the benchmark's A2 table
    reports, so "best on validation" and "best in the report" cannot diverge.
    """
    idx = bank.starts(horizon)
    if idx.numel() == 0:
        return float("nan")
    if idx.numel() > max_windows:
        idx = idx[torch.linspace(0, idx.numel() - 1, max_windows,
                                 device=idx.device).long()]
    tot, cnt = 0.0, 0
    for s in range(0, idx.numel(), chunk):
        sub = idx[s:s + chunk]
        nu_hat = rollout_windows(sim, bank, sub, horizon, n_sub, integrator)
        tgt = torch.stack([bank.NU[sub + j + 1] for j in range(horizon)], dim=1)
        e = ((nu_hat - tgt) * w_nu).pow(2)
        e = torch.nan_to_num(e, nan=1e6, posinf=1e6, neginf=1e6)
        tot += float(e.sum())
        cnt += e.numel()
    return float(np.sqrt(tot / max(cnt, 1)))
