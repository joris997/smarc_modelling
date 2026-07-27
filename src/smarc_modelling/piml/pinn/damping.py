#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cholesky-parametrised learned damping: ``D(x) = L(x) L(x)^T``, strictly PD.

Runtime module (imported by ``SAM_torch`` / ``SAM_PIML`` via ``pinn.load_pinn_D``); the
training scripts live in the submodule's ``train/``.

Motivation — three measured defects in the previous ``checkpoints/pinn.pt``
(12 -> 128x50 -> 36, ``D = A A^T``, 815,396 params):

1. **Cost.** 77.3 ms per forward at the planner's production batch (b=176,400, fp32,
   RTX 3500 Ada) => ~19.5 s per ``rollout_parallel`` against a 0.113 s white-box floor.
   50 *serial* layers is the problem, not the parameter count: nothing fuses.
2. **Only PSD, and singular.** Measured ``lambda_min(D) = -5.7e-7``.  An unconstrained
   ``A`` gives ``A A^T ⪰ 0`` but nothing keeps it away from 0.
3. **Stiff.** ``lambda_max(M^-1 D)`` reaches 175 s^-1, so forward Euler needs
   ``h < 0.011 s`` while the planner integrates at ``h = 0.02`` -- which is why
   ``main_sam.py`` had to switch to RK4 (4x the network evaluations).  The stiffness is
   not incidental: the regression target is dominated by white-box thrust/buoyancy error
   that no PD ``D`` can represent, and the only way a PD ``D`` can emit a near-constant
   force is to blow up as ``nu -> 0``.

What this module changes:

* **``L`` lower-triangular with a strictly positive diagonal** => ``det L > 0`` =>
  ``D = L L^T`` is *strictly* PD, not merely PSD.  21 free entries instead of 36.
* **Bounded spectrum by construction.**  The diagonal is a scaled ``sigmoid`` and the
  off-diagonal a scaled ``tanh``, so ``lambda_max(D) <= trace(D) = ||L||_F^2 <=
  lambda_cap`` for *any* input, saturating or adversarial.  ``pinn.pt`` has no such
  bound.  (The trace bound is ~3-6x loose against the true ``lambda_max``; it is a hard
  guarantee, not a tight one.  The Euler-stability *margin* is shaped by the training
  stiffness penalty and verified empirically -- see ``train/benchmark.py`` table A3.)
* **Fossen structure exactly.**  ``L = L_lin(u) + ||nu|| * L_quad(nu, u)``, so
  ``D = D_lin + D_quad(|nu|)`` holds by construction and ``D(nu=0) = L_lin L_lin^T`` is
  bounded -- the ``nu -> 0`` blow-up is structurally impossible.
* **Smooth.**  ``tanh``/``silu`` trunk, so ``D`` is C^inf.  A 50-layer ReLU net is only
  C^0: RK4 silently drops to first order across every kink, and the Jacobian jumps.
* **A force fast path.**  The dynamics only ever needs ``D @ nu_r``; with the Cholesky
  factor that is ``L @ (L^T nu_r)`` -- two triangular 6-vector matvecs done on the 21
  scalars, never materialising a ``(b,6,6)``.  Measured 1.702 ms -> 0.427 ms eager at
  b=176,400, and unlike ``bmm`` it fuses into the compiled ``_dyn``.
* **Decoupled clamp box and standardisation.**  ``pinn.pt`` used one min/max box for
  both, so a single mocap glitch (``p in [-757.9, 1.9]``) destroyed the *scaling* as well
  as the clamp -- p, q, r are effectively dead inputs in that model.  Here the clamp box
  is physical limits and the scaling is robust median/MAD of the training split.
"""
import numpy as np
import torch
import torch.nn as nn

#: Checkpoint discriminator.  Legacy ``pinn.pt`` files have no ``format`` key at all.
FORMAT = "chol_v1"

#: Row-major lower-triangular order: (0,0),(1,0),(1,1),(2,0),(2,1),(2,2),...,(5,5).
#: Pinned and round-trip tested -- a row/column-major mix-up here is silent and fatal.
TRIL_ORDER = "row_major"
TRIL_ROWS, TRIL_COLS = np.tril_indices(6)
#: Positions of the diagonal entries within the 21-vector.
DIAG_IDX = [0, 2, 5, 9, 14, 20]
OFFDIAG_IDX = [k for k in range(21) if k not in DIAG_IDX]
#: Row index of each off-diagonal entry, for the row-relative scaling in `l_flat`.
_OFFDIAG_ROW = [int(TRIL_ROWS[k]) for k in OFFDIAG_IDX]

#: The white-box constant damping (``SAM.dynamics`` overwrites its nonlinear D with this):
#: ``diag(damping_factor x3, damping_rot x3)``.
D_WHITEBOX = np.diag([60.0, 60.0, 60.0, 5.0, 5.0, 5.0])

#: Physical clamp box for the 12-D feature ``[u,v,w,p,q,r, vbs,lcg, dS,dR, rpm1,rpm2]``.
#: Deliberately NOT data statistics -- see the module docstring.  The velocity bounds
#: match ``train/quality.QualityConfig``'s rejection box; the actuator bounds are the
#: hardware limits (``SAM.Propellers.rpm_min/max``, VBS/LCG percent, fin travel).
CLAMP_LO = np.array([-0.8, -0.8, -0.8, -2.0, -2.0, -2.0,
                     0.0, 0.0, -0.13, -0.13, -1525.0, -1525.0])
CLAMP_HI = np.array([0.8, 0.8, 0.8, 2.0, 2.0, 2.0,
                     100.0, 100.0, 0.13, 0.13, 1525.0, 1525.0])

#: Largest ||nu|| the clamp box admits -- used to size the quadratic-drag caps.
NU_MAX = float(np.sqrt(3 * 0.8 ** 2 + 3 * 2.0 ** 2))

_ACTIVATIONS = {"tanh": torch.tanh, "silu": torch.nn.functional.silu,
                "relu": torch.relu, "gelu": torch.nn.functional.gelu}


def to_primitive(obj):
    """Recursively convert numpy types to plain Python ones.

    Checkpoints carry training metadata (config, history, split lists), and
    ``load_pinn_D`` reads with ``weights_only=True`` -- which rejects
    ``numpy._core.multiarray._reconstruct``.  A stray ``np.float64`` buried in a history
    dict is enough to make a checkpoint unloadable by its own loader, and the failure
    surfaces only at load time, long after training has finished.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_primitive(v) for v in obj]
    return obj


def _softplus_inv(y):
    return float(np.log(np.expm1(y)))


def _logit(p):
    return float(np.log(p / (1.0 - p)))


class CholeskyDamping(nn.Module):
    """``x (b,12) -> D (b,6,6)``, strictly PD, spectrum bounded, C^inf.

    The external contract is identical to ``PINN_D``: ``forward`` takes the raw 12-D
    ``concat(nu, u_ref)`` and returns a ``(b,6,6)``.  ``damping_force`` is the fast path
    the batched dynamics uses; ``L`` and ``l_flat`` exist for tests and diagnostics.
    """

    def __init__(self, in_dim=12, hidden=32, n_hidden=2, activation="tanh",
                 clamp_lo=None, clamp_hi=None, x_mu=None, x_sigma=None,
                 d_floor=None, d_ceil=None, o_ceil=0.08,
                 dq_ceil=None, oq_ceil=0.01, fossen_split=True):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}")
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_hidden = n_hidden
        self.activation_name = activation
        self.act = _ACTIVATIONS[activation]
        self.fossen_split = bool(fossen_split)
        self.out_dim = 42 if self.fossen_split else 21

        clamp_lo = CLAMP_LO if clamp_lo is None else clamp_lo
        clamp_hi = CLAMP_HI if clamp_hi is None else clamp_hi
        # Default standardisation = the clamp box mapped to [-1, 1]; `from_checkpoint`
        # and the trainer overwrite it with robust median / MAD of the training split.
        if x_mu is None:
            x_mu = 0.5 * (np.asarray(clamp_lo) + np.asarray(clamp_hi))
        if x_sigma is None:
            x_sigma = 0.5 * (np.asarray(clamp_hi) - np.asarray(clamp_lo))

        def buf(name, v):
            self.register_buffer(name, torch.as_tensor(np.asarray(v), dtype=torch.float32),
                                 persistent=False)

        buf("x_min", np.asarray(clamp_lo))
        buf("x_range", np.asarray(clamp_hi) - np.asarray(clamp_lo))   # legacy-compatible
        buf("x_mu", x_mu)
        buf("x_sigma", np.maximum(np.asarray(x_sigma), 1e-3))

        # --- output-head caps -------------------------------------------------
        # L_lin's diagonal is strictly positive (that is what makes D strictly PD);
        # L_quad's may vanish (a DOF need not have quadratic drag).  Defaults let the
        # model reach ~2x the white-box damping per DOF.
        # Sized so the row-sum bound in `lambda_cap` lands near 240, i.e.
        # lambda_max(M^-1 D) <= 240 / lambda_min(M) ~ 69 s^-1 => Euler h_stab ~ 0.029 s,
        # a ~1.4x margin over the planner's h = dt/n_integrator = 0.02 s.  The linear
        # diagonal reaches D_ii = 100 (1.7x the white box's 60) and the quadratic part
        # adds at most ~30% more at the corner of the clamp box.
        d_floor = np.full(6, 0.5) if d_floor is None else np.asarray(d_floor, float)
        d_ceil = np.array([10.0, 10.0, 10.0, 3.0, 3.0, 3.0]) if d_ceil is None \
            else np.asarray(d_ceil, float)
        dq_ceil = np.array([0.8, 0.8, 0.8, 0.24, 0.24, 0.24]) if dq_ceil is None \
            else np.asarray(dq_ceil, float)
        buf("d_floor", d_floor)
        buf("d_ceil", d_ceil)
        buf("dq_ceil", dq_ceil)
        # Off-diagonals are a FRACTION OF THEIR OWN ROW'S DIAGONAL, not absolute:
        #   L = diag(d) (I + N),  N strictly lower,  |N_ij| <= o_tot.
        # A diagonal floor alone bounds det(L) but not the CONDITIONING of L -- with
        # absolute off-diagonals of order 1 against a floor of 0.05, L goes near-singular
        # and fp32 `eigvalsh(D)` returns small negatives (measured -8.5e-6).  Row scaling
        # makes `I + N` strictly diagonally dominant whenever `5 * o_tot < 1`, which
        # turns lambda_min into a construction guarantee as well -- see `lambda_min_bound`.
        self.o_ceil = float(o_ceil)
        self.oq_ceil = float(oq_ceil)
        if 5.0 * (self.o_ceil + NU_MAX * self.oq_ceil) >= 1.0:
            raise ValueError(
                f"o_ceil + NU_MAX*oq_ceil = {self.o_ceil + NU_MAX * self.oq_ceil:.3f} "
                f"must be < 0.2 to keep (I + N) diagonally dominant, i.e. L invertible "
                f"with a computable margin.")

        # --- trunk ------------------------------------------------------------
        # Feature augmentation (below) triples the nu block: [z, |z_nu|, z_nu*|z_nu|].
        # |nu| is exactly Fossen's quadratic-drag regressor, so a linear head on it
        # already reproduces D_quad.
        feat_dim = in_dim + 12
        self.layers = nn.ModuleList()
        prev = feat_dim
        for _ in range(n_hidden):
            self.layers.append(nn.Linear(prev, hidden))
            prev = hidden
        self.output_layer = nn.Linear(prev, self.out_dim)
        self.reset_to_whitebox()

    # ------------------------------------------------------------------
    @property
    def o_tot(self):
        """Bound on ``|N_ij|``, the row-relative off-diagonal magnitude."""
        return self.o_ceil + (NU_MAX * self.oq_ceil if self.fossen_split else 0.0)

    def d_max(self):
        """(6,) upper bound on the diagonal ``L_ii`` anywhere in the clamp box."""
        d = self.d_ceil.cpu().numpy().copy()
        if self.fossen_split:
            # ||nu|| over the clamp box is at most NU_MAX, and sigmoid is bounded by 1.
            d = d + NU_MAX * self.dq_ceil.cpu().numpy()
        return d

    def l_abs_cap(self):
        """(6,6) elementwise upper bound on ``|L_ij|``, valid anywhere in the clamp box."""
        d = self.d_max()
        cap = np.tril(np.outer(d, np.ones(6)) * self.o_tot)
        np.fill_diagonal(cap, d)
        return cap

    @property
    def lambda_min_bound(self):
        """Hard lower bound on ``lambda_min(D) = sigma_min(L)^2 > 0``.

        ``L = diag(d)(I + N)`` so ``sigma_min(L) >= min(d) * sigma_min(I + N)``, and for
        strictly lower-triangular ``N`` with ``||N||_inf <= 5 o_tot < 1``,
        ``sigma_min(I+N) >= (1 - 5 o_tot) / sqrt(6)``.
        """
        s = (1.0 - 5.0 * self.o_tot) / np.sqrt(6.0)
        return float((self.d_floor.cpu().numpy().min() * s) ** 2)

    @property
    def lambda_cap(self):
        """Hard upper bound on ``lambda_max(D)`` over the whole clamp box.

        For a symmetric ``D``, ``lambda_max(D) <= ||D||_inf = max_i sum_j |D_ij|``, and
        ``|D_ij| = |sum_k L_ik L_jk| <= sum_k cap_ik cap_jk``.  This row-sum bound is
        several times tighter than the obvious ``lambda_max <= trace(D) = ||L||_F^2``
        (which sums all six eigenvalues and so is ~6x loose for a near-diagonal D):
        measured on the default caps, 240 vs 549.

        The bound holds for ANY input, including saturating ones -- it depends only on
        the output-head caps, not on the weights.  It is what makes "the learned D cannot
        blow up as nu -> 0" a construction guarantee rather than a training outcome.
        """
        cap = self.l_abs_cap()
        return float((cap @ cap.sum(axis=0)).max())

    def reset_to_whitebox(self):
        """Zero the head's weight and set its bias so the untrained net *is* the white box.

        Two things this buys: Stage-B rollouts are stable from step 1 (no NaN-poisoned
        early epochs), and "beat the white-box baseline" is monotone from the start
        rather than something training has to claw back to.
        """
        nn.init.zeros_(self.output_layer.weight)
        with torch.no_grad():
            b = torch.zeros(self.out_dim)
            lo = self.d_floor.cpu().numpy()
            hi = self.d_ceil.cpu().numpy()
            l_wb = np.sqrt(np.diag(D_WHITEBOX))            # chol of a diagonal matrix
            for i, k in enumerate(DIAG_IDX):
                frac = np.clip((l_wb[i] - lo[i]) / (hi[i] - lo[i]), 1e-4, 1 - 1e-4)
                b[k] = _logit(float(frac))
            for k in OFFDIAG_IDX:
                b[k] = 0.0                                  # tanh(0) = 0
            if self.fossen_split:
                # sigmoid(-8) ~ 3e-4: the quadratic part starts at essentially zero.
                b[21 + np.array(DIAG_IDX)] = -8.0
                b[21 + np.array(OFFDIAG_IDX)] = 0.0
            self.output_layer.bias.copy_(b)

    # ------------------------------------------------------------------
    def _features(self, x):
        """Clamp to the physical box, standardise, then augment with |nu| terms."""
        x = torch.clamp(x, self.x_min, self.x_min + self.x_range)
        z = (x - self.x_mu) / self.x_sigma
        zn = z[:, :6]
        a = torch.abs(zn)
        return torch.cat([z, a, zn * a], dim=1)

    def l_flat(self, x):
        """``(b,12) -> (b,21)`` row-major lower-triangular entries of ``L``.

        All the shared work lives here; ``forward`` / ``L`` / ``damping_force`` differ
        only in what they do with the 21 numbers.
        """
        h = self._features(x)
        for layer in self.layers:
            h = self.act(layer(h))
        raw = self.output_layer(h)

        d_floor = self.d_floor.to(raw.dtype)
        d_ceil = self.d_ceil.to(raw.dtype)
        # Diagonal: strictly positive, hard-capped.   Off-diagonal: bounded, row-relative.
        diag = d_floor + (d_ceil - d_floor) * torch.sigmoid(raw[:, DIAG_IDX])
        n_off = self.o_ceil * torch.tanh(raw[:, OFFDIAG_IDX])

        if self.fossen_split:
            rq = raw[:, 21:]
            # ||nu|| from the CLAMPED velocities, so `lambda_cap` really holds.
            nu_c = torch.clamp(x[:, :6], self.x_min[:6], (self.x_min + self.x_range)[:6])
            speed = torch.linalg.vector_norm(nu_c, dim=1, keepdim=True)
            # L = L_lin + ||nu|| L_quad  <=>  Fossen's D = D_lin + D_quad(|nu|), exactly.
            diag = diag + speed * self.dq_ceil.to(raw.dtype) * torch.sigmoid(rq[:, DIAG_IDX])
            n_off = n_off + speed * self.oq_ceil * torch.tanh(rq[:, OFFDIAG_IDX])

        out = raw.new_empty(raw.shape[0], 21)
        out[:, DIAG_IDX] = diag
        # Scale each off-diagonal by ITS OWN ROW's diagonal: L = diag(d) (I + N).
        out[:, OFFDIAG_IDX] = n_off * diag[:, _OFFDIAG_ROW]
        return out

    def L(self, x):
        """``(b,12) -> (b,6,6)`` lower-triangular Cholesky factor (dense, for tests)."""
        lf = self.l_flat(x)
        out = lf.new_zeros(lf.shape[0], 6, 6)
        out[:, TRIL_ROWS, TRIL_COLS] = lf
        return out

    def forward(self, x):
        """``(b,12) -> (b,6,6)`` damping matrix.  Same contract as ``PINN_D.forward``."""
        Lm = self.L(x)
        return Lm @ Lm.transpose(-2, -1)

    def damping_force(self, x, nu_r):
        """``D(x) @ nu_r`` as ``L @ (L^T nu_r)`` -- no ``(b,6,6)``, no ``bmm``.

        Written over the 21 scalar columns so every op is elementwise and
        ``torch.compile`` folds the whole thing into ``SAM_torch._dyn``.
        """
        l = self.l_flat(x).to(nu_r.dtype)
        n0, n1, n2, n3, n4, n5 = nu_r.unbind(1)
        (l00, l10, l11, l20, l21, l22, l30, l31, l32, l33,
         l40, l41, l42, l43, l44, l50, l51, l52, l53, l54, l55) = l.unbind(1)

        # y = L^T nu
        y0 = l00 * n0 + l10 * n1 + l20 * n2 + l30 * n3 + l40 * n4 + l50 * n5
        y1 = l11 * n1 + l21 * n2 + l31 * n3 + l41 * n4 + l51 * n5
        y2 = l22 * n2 + l32 * n3 + l42 * n4 + l52 * n5
        y3 = l33 * n3 + l43 * n4 + l53 * n5
        y4 = l44 * n4 + l54 * n5
        y5 = l55 * n5

        # z = L y
        z0 = l00 * y0
        z1 = l10 * y0 + l11 * y1
        z2 = l20 * y0 + l21 * y1 + l22 * y2
        z3 = l30 * y0 + l31 * y1 + l32 * y2 + l33 * y3
        z4 = l40 * y0 + l41 * y1 + l42 * y2 + l43 * y3 + l44 * y4
        z5 = l50 * y0 + l51 * y1 + l52 * y2 + l53 * y3 + l54 * y4 + l55 * y5
        return torch.stack([z0, z1, z2, z3, z4, z5], dim=1)

    # ------------------------------------------------------------------
    def arch(self):
        return {"in_dim": self.in_dim, "hidden": self.hidden, "n_hidden": self.n_hidden,
                "out_dim": self.out_dim, "activation": self.activation_name,
                "fossen_split": self.fossen_split, "tril_order": TRIL_ORDER}

    def checkpoint(self, **extra):
        """Serializable dict in the ``chol_v1`` format.

        Buffers are stored as torch TENSORS, not numpy arrays: ``load_pinn_D`` reads with
        ``weights_only=True`` (the torch>=2.6 default), which rejects
        ``numpy._core.multiarray._reconstruct``.  A numpy payload here makes the
        checkpoint unloadable by its own loader.
        """
        ck = {
            "format": FORMAT,
            "arch": self.arch(),
            "state_dict": self.state_dict(),
            "x_min": self.x_min.detach().cpu(),
            "x_range": self.x_range.detach().cpu(),
            "x_mu": self.x_mu.detach().cpu(),
            "x_sigma": self.x_sigma.detach().cpu(),
            "d_floor": self.d_floor.detach().cpu(),
            "d_ceil": self.d_ceil.detach().cpu(),
            "dq_ceil": self.dq_ceil.detach().cpu(),
            "o_ceil": self.o_ceil,
            "oq_ceil": self.oq_ceil,
            "lambda_cap": self.lambda_cap,
            "lambda_min_bound": self.lambda_min_bound,
            "quat_order": "wxyz",
        }
        ck.update(to_primitive(extra))
        return ck

    @classmethod
    def from_checkpoint(cls, ck):
        a = ck["arch"]
        lo = np.asarray(ck["x_min"], dtype=float)
        model = cls(
            in_dim=a["in_dim"], hidden=a["hidden"], n_hidden=a["n_hidden"],
            activation=a.get("activation", "tanh"),
            fossen_split=a.get("fossen_split", True),
            clamp_lo=lo, clamp_hi=lo + np.asarray(ck["x_range"], dtype=float),
            x_mu=np.asarray(ck["x_mu"], dtype=float),
            x_sigma=np.asarray(ck["x_sigma"], dtype=float),
            d_floor=np.asarray(ck["d_floor"], dtype=float),
            d_ceil=np.asarray(ck["d_ceil"], dtype=float),
            dq_ceil=np.asarray(ck["dq_ceil"], dtype=float),
            o_ceil=float(ck["o_ceil"]), oq_ceil=float(ck["oq_ceil"]),
        )
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return model
