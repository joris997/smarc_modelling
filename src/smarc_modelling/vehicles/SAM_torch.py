#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM_torch.py:

   Batched / vectorised PyTorch port of the SAM AUV dynamics in ``SAM.py``.

   ``SAM.dynamics(x, u_ref)`` integrates ONE 19-D state with NumPy.  The
   Graphs-of-Bundles planner needs the same ``dx = f(x, u)`` evaluated for a whole
   *batch* of (sample, region) states at once — see ``classes/robots/sam.py``,
   whose ``rollout_parallel`` otherwise loops over every knot, sample, region and
   Euler substep in Python.  ``SAMTorch`` reproduces the same physics but with a
   leading batch dimension ``b``, so the whole bundle is one tensor op.

   State reduction (vs the NumPy ``SAM.py``): this port uses a **15-D** state
   ``[eta(7) | nu(6) | act(2)=[vbs, lcg]]``.  The fin/rpm actuator states of the
   full 19-D model are dropped — they never drive forces through the *state* (the
   propeller reads the command ``U_ref`` directly), so carrying them only added
   trivial consensus dims to the GCS.  ``vbs``/``lcg`` are kept as proper
   integrating states (they set buoyancy mass / CoG / inertia).

   The public surface is:

       SAMTorch.dynamics(X, U_ref) -> dX          # X:(b,15) U_ref:(b,6) -> (b,15)
       SAMTorch.euler_rollout(X0, U_ref_traces, duration, n_euler) -> X  # (b,15)

   Both accept NumPy arrays (or torch tensors) and return NumPy arrays.

   Parity notes — the eta/nu/[vbs,lcg] physics is a faithful re-implementation of
   ``SAM.dynamics`` (originally validated row-by-row to < 1e-8; the 15-D reduction
   only drops the fin/rpm state derivatives — see
   ``classes/robots/smarc_modelling/test/test_sam_torch_parity.py``):

   * The elaborate nonlinear ``SAM.calculate_D`` is **dead code**: ``SAM.dynamics``
     overwrites ``self.D`` with a constant diagonal right after calling it, so this
     port skips the nonlinear damping entirely and uses that constant ``D``.
     Passing ``piml_type="pinn"`` instead swaps in the learned, state-dependent
     ``D(nu, u)`` from ``SAM_PIML`` (a batched PINN forward pass); the rest of the
     physics is unchanged.
   * Buoyancy ``B`` is constant (set once at init); only weight ``W = m*g`` varies
     with the VBS mass.
   * All scalar/geometry constants are *lifted* from a reference NumPy ``SAM``
     instance so the two models share identical parameters (k-factors, prop
     coefficients, CG/CB offsets, masses, ...).

Author: ported for the bundle-stl Graphs-of-Bundles pipeline.
"""

import math

import numpy as np
import torch

from smarc_modelling.vehicles.SAM import SAM as _SAMNumpy


def _skew_np(v):
    """3x3 skew-symmetric matrix (NumPy, used only for constant pre-computation)."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


class SAMTorch:
    """Batched PyTorch SAM dynamics mirroring ``smarc_modelling.vehicles.SAM.SAM``.

    Constants are lifted from a reference NumPy ``SAM`` so the physics match exactly.
    Everything that depends on the state (mass from VBS fill, inertia, Coriolis,
    restoring forces, propeller thrust) is recomputed per call, fully vectorised over
    a leading batch dimension ``b``.
    """

    def __init__(self, dt=0.02, V_current=0.0, beta_current=0.0,
                 device="auto", dtype=torch.float64, piml_type=None,
                 tau_act=None, compile_mode=None, piml_ckpt=None,
                 piml_ckpt_name="pinn.pt", piml_model=None):
        # Reference NumPy model: single source of truth for every constant.
        ref = _SAMNumpy(dt=dt, V_current=V_current, beta_current=beta_current)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        # `dt` is the nominal integrator substep.  It is NEVER reassigned: the rollouts
        # take their step size as an argument.  (It used to be overwritten with a
        # per-row tensor, which left a stale `(b_prev, 1)` on the instance and made a
        # later call at a different batch size die in `torch.cat`.)
        self.dt = float(dt)
        # Actuator (VBS/LCG) first-order tracking time constant.  This used to be the
        # integration substep, which made the ACTUATOR MODEL depend on the
        # discretization -- and, once each bundle sample integrates its own horizon,
        # depend on the sample.  Defaulting to `dt` reproduces the old numbers exactly
        # at the nominal step while making `_dyn` a pure f(x, u).
        self.tau_act = float(dt if tau_act is None else tau_act)

        def t(arr):
            return torch.as_tensor(np.asarray(arr, dtype=np.float64),
                                   dtype=self.dtype, device=self.device)

        # --- scalar constants -------------------------------------------------
        self.g = float(ref.g)
        self.rho = float(ref.rho)
        self.rho_w = float(ref.rho_w)
        self.B = float(ref.B)                 # constant buoyancy (W at init)
        self.gamma = float(ref.gamma)

        self.m_ss = float(ref.ss.m_ss)
        self.m_lcg = float(ref.lcg.m_lcg)
        self.r_vbs = float(ref.vbs.r_vbs)
        self.l_vbs_l = float(ref.vbs.l_vbs_l)
        self.l_lcg_l = float(ref.lcg.l_lcg_l)
        self.h_lcg_dim = float(ref.lcg.h_lcg_dim)

        self.k1 = float(ref.k1)
        self.k2 = float(ref.k2)
        self.k_prime = float(ref.k_prime)
        self.r44 = float(ref.r44)

        self.inertia_factor = float(ref.inertia_factor)
        self.damping_factor = float(ref.damping_factor)
        self.damping_rot = float(ref.damping_rot)
        self.thruster_rot_strength = float(ref.thruster_rot_strength)

        # propeller coefficients
        self.D_prop = float(ref.D_prop)
        self.Va_coef = float(ref.Va_coef)
        self.KT_0 = float(ref.KT_0)
        self.KQ_0 = float(ref.KQ_0)
        self.KT_max = float(ref.KT_max)
        self.KQ_max = float(ref.KQ_max)
        self.Ja_max = float(ref.Ja_max)

        # current
        self.V_c = float(ref.V_c)
        self.beta_c = float(ref.beta_c)

        # actuator rate limits (symmetric, as used in SAM.actuator_dynamics)
        self.vbs_dot_max = float(ref.vbs.x_vbs_dot_max)
        self.lcg_dot_max = float(ref.lcg.x_lcg_dot_max)

        # --- constant geometry vectors ---------------------------------------
        p_OSsg_O = np.asarray(ref.ss.p_OSsg_O, dtype=np.float64)
        p_OVbs_O = np.asarray(ref.vbs.p_OVbs_O, dtype=np.float64)
        p_OLcgPos_O = np.asarray(ref.lcg.p_OLcgPos_O, dtype=np.float64)
        p_OC_O = np.asarray(ref.p_OC_O, dtype=np.float64)
        p_OB_O = np.asarray(ref.p_OB_O, dtype=np.float64)

        self.p_OSsg_O = t(p_OSsg_O)            # (3,)
        self.p_OVbs_O = t(p_OVbs_O)            # (3,)
        self.p_OLcgPos_O = t(p_OLcgPos_O)      # (3,)
        self.p_OC_O = t(p_OC_O)                # (3,)
        self.p_OB_O = t(p_OB_O)                # (3,)

        # propeller shaft locations (list of 2 vectors)
        self.r_t_p_sh = [t(np.asarray(r, dtype=np.float64))
                         for r in ref.propellers.r_t_p_sh]

        # semi-axes for the solid-structure inertia
        a = float(ref.a)
        b = float(ref.b)

        # --- constant inertia pieces -----------------------------------------
        # Solid structure (entirely constant -> J_ss_co).
        Ix_ss = (2.0 / 5.0) * self.m_ss * b ** 2
        Iy_ss = (1.0 / 5.0) * self.m_ss * (a ** 2 + b ** 2)
        Iz_ss = Iy_ss
        J_ss_cg = np.diag([Ix_ss, Iy_ss, Iz_ss])
        S2_ss = _skew_np(p_OSsg_O) @ _skew_np(p_OSsg_O)
        J_ss_co = J_ss_cg - self.m_ss * S2_ss
        self.J_ss_co = t(J_ss_co)              # (3,3)

        # VBS squared-skew (constant; mass/inertia magnitude varies per batch).
        self.S2_vbs = t(_skew_np(p_OVbs_O) @ _skew_np(p_OVbs_O))   # (3,3)

        # LCG center-of-gravity inertia (constant); its squared-skew varies because
        # p_OLcg_O depends on the LCG actuator position.
        Ix_lcg = (1.0 / 2.0) * self.m_lcg * (self.h_lcg_dim / 2.0) ** 2
        Iy_lcg = (1.0 / 12.0) * self.m_lcg * (3.0 * (self.h_lcg_dim / 2.0) ** 2
                                              + self.l_lcg_l ** 2)
        Iz_lcg = Iy_lcg
        self.J_lcg_cg = t(np.diag([Ix_lcg, Iy_lcg, Iz_lcg]))       # (3,3)

        # --- constant damping matrix (the overwrite in SAM.dynamics) ----------
        D = np.eye(6) * self.damping_factor
        D[3, 3] = self.damping_rot
        D[4, 4] = self.damping_rot
        D[5, 5] = self.damping_rot
        self.D = t(D)                          # (6,6)

        # --- optional learned damping (PINN) ---------------------------------
        # ``piml_type="pinn"`` swaps the constant ``self.D`` above for a learned,
        # state-dependent 6x6 ``D`` predicted per batch row from (nu, u) -- the same
        # network the NumPy ``SAM_PIML`` uses (see piml/pinn/pinn.py).  The network
        # is batched natively, so the whole bundle is one forward pass; it is cast to
        # this model's device/dtype so it composes with the float64 dynamics.
        self.piml_type = piml_type
        self.piml_model = None
        if self.piml_type == "pinn":
            # `piml_ckpt` overrides the search for `piml_ckpt_name` under checkpoints/,
            # which is git-ignored (the trained weights are a local artefact, not source).
            # `piml_model` short-circuits loading entirely -- that is how the trainer
            # hands in the live, gradient-carrying model (see train/rollout.py).
            if piml_model is not None:
                self.piml_model = piml_model
            else:
                from smarc_modelling.piml.pinn.pinn import load_pinn_D
                self.piml_model = load_pinn_D(piml_ckpt, piml_ckpt_name)
            self.piml_model = self.piml_model.to(device=self.device, dtype=self.dtype)
        self._bind_damping_force()

        self._pi = math.pi

        # ------------------------------------------------------------------
        # Scalarized constants for the fused `_dyn`.
        #
        # `_dyn` is bandwidth bound, so it is written over SCALAR columns and never
        # allocates a (b,3,3)/(b,6,6).  Its constants are therefore kept as plain
        # PYTHON FLOATS, not tensors, for three reasons:
        #   * Inductor bakes them into the kernel as immediate operands (no loads), and
        #     structural zeros constant-fold away for free.
        #   * Python floats are *weak* scalars in torch, so the identical source runs
        #     in float32 and float64 with no `.to(dtype)` anywhere — which is what makes
        #     the dtype knob a one-line change.
        #   * Nothing for Dynamo to guard on.
        # The tensor copies above are kept: the reference helpers still use them, and
        # they are the parity oracle for what follows.
        # ------------------------------------------------------------------
        self._pSsg = tuple(float(v) for v in p_OSsg_O)
        self._pVbs = tuple(float(v) for v in p_OVbs_O)
        self._pLcg0 = tuple(float(v) for v in p_OLcgPos_O)
        self._pOC = tuple(float(v) for v in p_OC_O)
        self._pOB = tuple(float(v) for v in p_OB_O)
        self._rsh = [tuple(float(v) for v in np.asarray(r, dtype=np.float64))
                     for r in ref.propellers.r_t_p_sh]

        # J_ss_co + J_lcg_cg, summed once (both constant); upper triangle only, since
        # every term here is symmetric by construction.
        Jc = np.asarray(J_ss_co, dtype=np.float64) + np.diag([Ix_lcg, Iy_lcg, Iz_lcg])
        self._Jc00, self._Jc01, self._Jc02 = float(Jc[0, 0]), float(Jc[0, 1]), float(Jc[0, 2])
        self._Jc11, self._Jc12, self._Jc22 = float(Jc[1, 1]), float(Jc[1, 2]), float(Jc[2, 2])

        S2v = _skew_np(p_OVbs_O) @ _skew_np(p_OVbs_O)
        self._S2v00, self._S2v01, self._S2v02 = float(S2v[0, 0]), float(S2v[0, 1]), float(S2v[0, 2])
        self._S2v11, self._S2v12, self._S2v22 = float(S2v[1, 1]), float(S2v[1, 2]), float(S2v[2, 2])

        # VBS mass per unit fill-percent, and the VBS length per percent.
        self._c_mvbs = self.rho_w * math.pi * self.r_vbs ** 2 * (self.l_vbs_l / 100.0)
        self._c_xvbs = self.l_vbs_l / 100.0
        self._c_xlcg = self.l_lcg_l / 100.0

        # Pre-multiplied propeller coefficients.
        self._cT = self.rho * self.D_prop ** 4
        self._cQ = self.rho * self.D_prop ** 5
        # Advance-ratio slopes.  The /D_prop of the reference's `Va / D_prop` lives in
        # `Va_over_D` at the call site, NOT here.
        self._ktsl = (self.KT_max - self.KT_0) / self.Ja_max
        self._kqsl = (self.KQ_max - self.KQ_0) / self.Ja_max

        # Constant diagonal damping (the `calculate_D` overwrite in SAM.dynamics).
        self._Dlin = self.damping_factor
        self._Drot = self.damping_rot

        # ------------------------------------------------------------------
        # Compiled substeps.
        #
        # Measured on an RTX 3500 Ada at the planner's batch size, per `_dyn` call:
        # fp64 3.03 -> 0.74 ms (4.1x), fp32 1.91 -> 0.12 ms (15.6x); at b=49 (the
        # `rollout_onestep` shape, which is pure launch overhead) 1.8 -> 0.06 ms (33x).
        # `dynamic=False` gives one graph per batch size — there are only a handful
        # (1, N-1, N1*Ms*K) and each compiles in ~5 s, once per process.
        #
        # NOT `mode="reduce-overhead"` by default: cudagraph-trees plus a loop that
        # feeds each step's output into the next is a known source of silent
        # output-overwrite bugs.  It is available via `compile_mode` for benchmarking.
        # ------------------------------------------------------------------
        self.compile_mode = compile_mode
        if compile_mode is None:
            self._step_euler_c = self._step_euler
            self._step_rk4_c = self._step_rk4
        else:
            # A recompile storm is a real failure mode here, so give Dynamo headroom
            # and let each distinct batch size specialize.
            torch._dynamo.config.cache_size_limit = max(
                32, torch._dynamo.config.cache_size_limit)
            kw = {} if compile_mode == "default" else {"mode": compile_mode}
            self._step_euler_c = torch.compile(self._step_euler, dynamic=False, **kw)
            self._step_rk4_c = torch.compile(self._step_rk4, dynamic=False, **kw)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _bind_damping_force(self):
        """Pick how the learned damping force ``D(x) @ nu_r`` is evaluated.

        The dynamics only ever needs the PRODUCT, never ``D`` itself.  A Cholesky model
        (``D = L L^T``) can deliver it as ``L @ (L^T nu_r)`` -- two triangular 6-vector
        matvecs, 36 elementwise ops that ``torch.compile`` fuses straight into ``_dyn``.
        Materialising the ``(b,6,6)`` and calling ``bmm`` instead costs a measured
        1.702 ms vs 0.427 ms at b=176400 (eager), and the ``bmm`` can never fuse.

        Models without a ``damping_force`` (the legacy full-A ``PINN_D``) fall back to
        the dense path, so they keep working untouched.  Call this again after
        reassigning ``self.piml_model``.
        """
        m = self.piml_model
        if m is None:
            self._damping_force = None
        elif hasattr(m, "damping_force"):
            self._damping_force = m.damping_force
        else:
            def _dense(feat, nu_r):
                return torch.bmm(m(feat), nu_r.unsqueeze(-1)).squeeze(-1)
            self._damping_force = _dense

    def _t(self, arr):
        """To this model's device/dtype.  Passes tensors through without a host round-trip."""
        if torch.is_tensor(arr):
            return arr.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.asarray(arr, dtype=np.float64),
                               dtype=self.dtype, device=self.device)

    def _skew(self, v):
        """Batched skew-symmetric matrix.  v:(b,3) -> (b,3,3)."""
        b = v.shape[0]
        S = torch.zeros(b, 3, 3, dtype=self.dtype, device=self.device)
        S[:, 0, 1] = -v[:, 2]
        S[:, 0, 2] = v[:, 1]
        S[:, 1, 0] = v[:, 2]
        S[:, 1, 2] = -v[:, 0]
        S[:, 2, 0] = -v[:, 1]
        S[:, 2, 1] = v[:, 0]
        return S

    def _quat_to_dcm(self, q):
        """Unit-quaternion (scalar-first [q0,q1,q2,q3]) -> DCM (b,3,3).

        Matches ``scipy ... Rotation.from_quat(scalar_last).as_matrix()`` used by
        ``gnc.quaternion_to_dcm`` (Hamilton / active convention).
        """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        b = q.shape[0]
        R = torch.empty(b, 3, 3, dtype=self.dtype, device=self.device)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    @staticmethod
    def _angles_from_dcm(R):
        """Extract (psi, theta, phi) from a DCM the same way scipy ``as_euler('xyz')``
        does, i.e. decomposing R = Rz(psi) Ry(theta) Rx(phi).
        """
        theta = torch.asin(torch.clamp(-R[:, 2, 0], -1.0, 1.0))
        phi = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        psi = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        return psi, theta, phi

    def _m2c(self, M, nu):
        """Batched Coriolis matrix (6-DOF), mirroring ``gnc.m2c``.  M:(b,6,6) nu:(b,6)."""
        Msym = 0.5 * (M + M.transpose(-1, -2))
        M11 = Msym[:, 0:3, 0:3]
        M12 = Msym[:, 0:3, 3:6]
        M21 = M12.transpose(-1, -2)
        M22 = Msym[:, 3:6, 3:6]
        nu1 = nu[:, 0:3].unsqueeze(-1)
        nu2 = nu[:, 3:6].unsqueeze(-1)
        dt_dnu1 = (torch.bmm(M11, nu1) + torch.bmm(M12, nu2)).squeeze(-1)
        dt_dnu2 = (torch.bmm(M21, nu1) + torch.bmm(M22, nu2)).squeeze(-1)

        b = M.shape[0]
        C = torch.zeros(b, 6, 6, dtype=self.dtype, device=self.device)
        S1 = self._skew(dt_dnu1)
        S2 = self._skew(dt_dnu2)
        C[:, 0:3, 3:6] = -S1
        C[:, 3:6, 0:3] = -S1
        C[:, 3:6, 3:6] = -S2
        return C

    def _bound_actuators(self, u):
        """Clamp the VBS/LCG commands to [0, 100] (matches ``SAM.bound_actuators``)."""
        u = u.clone()
        u[:, 0] = torch.clamp(u[:, 0], 0.0, 100.0)
        u[:, 1] = torch.clamp(u[:, 1], 0.0, 100.0)
        return u

    def _propeller_force(self, U_speed, u_ref):
        """Batched propeller force/torque, mirroring ``SAM.calculate_propeller_force``.

        U_speed:(b,) total speed magnitude, u_ref:(b,6).  Returns tau:(b,6).
        """
        b = u_ref.shape[0]
        delta_s = -u_ref[:, 2]
        delta_r = -u_ref[:, 3]
        n_rpm = u_ref[:, 4:6]
        n_rps = n_rpm / 60.0
        Va = self.Va_coef * U_speed

        # C_T2C = Rz(delta_r) @ Ry(delta_s)   (calculate_dcm(order=[2,3], ...))
        cds, sds = torch.cos(delta_s), torch.sin(delta_s)
        cdr, sdr = torch.cos(delta_r), torch.sin(delta_r)
        zero = torch.zeros_like(cds)
        one = torch.ones_like(cds)
        Ry = torch.stack([
            torch.stack([cds, zero, -sds], dim=1),
            torch.stack([zero, one, zero], dim=1),
            torch.stack([sds, zero, cds], dim=1),
        ], dim=1)
        Rz = torch.stack([
            torch.stack([cdr, sdr, zero], dim=1),
            torch.stack([-sdr, cdr, zero], dim=1),
            torch.stack([zero, zero, one], dim=1),
        ], dim=1)
        C_T2C = torch.bmm(Rz, Ry)              # (b,3,3)

        tau = torch.zeros(b, 6, dtype=self.dtype, device=self.device)
        for i in range(n_rps.shape[1]):
            n = n_rps[:, i]
            abs_n = torch.abs(n)
            pos = n > 0

            X_pos = self.rho * (self.D_prop ** 4) * (
                self.KT_0 * abs_n * n
                + (self.KT_max - self.KT_0) / self.Ja_max * (Va / self.D_prop) * abs_n)
            X_neg = self.rho * (self.D_prop ** 4) * (self.KT_0 * abs_n * n) / 10.0
            X_prop = torch.where(pos, X_pos, X_neg)

            K_pos = self.rho * (self.D_prop ** 5) * (
                self.KQ_0 * abs_n * n
                + (self.KQ_max - self.KQ_0) / self.Ja_max * (Va / self.D_prop) * abs_n)
            K_neg = self.rho * (self.D_prop ** 5) * self.KQ_0 * abs_n * n / 10.0
            K_prop = torch.where(pos, K_pos, K_neg)

            dir_flip = torch.where(pos, torch.ones_like(n), -torch.ones_like(n))

            # F_prop_b = C_T2C @ [X, 0, 0] = X * first column of C_T2C
            F_prop_b = X_prop.unsqueeze(1) * C_T2C[:, :, 0]        # (b,3)

            r_sh = self.r_t_p_sh[i].view(1, 3, 1)
            r_prop = torch.bmm(C_T2C, r_sh.expand(b, 3, 1)).squeeze(-1) - self.p_OC_O

            base = torch.cross(r_prop, F_prop_b, dim=1)            # (b,3)
            sign_i = 1.0 if (i % 2 == 0) else -1.0
            base = base.clone()
            base[:, 0] = base[:, 0] + sign_i * K_prop

            s = self.thruster_rot_strength
            # base = r x F is already in body [roll, pitch, yaw] order, and K_prop
            # (axial torque) was added to base[0] (roll).  The moment slots of tau are
            # [.., roll(p), pitch(q), yaw(r)], so map straight through — no axis swap.
            # (The previous [base2, base1, base0] reorder put the rudder's large
            # lever-arm yaw moment onto roll and left yaw nearly inert; see
            # test/test_sam_turn.py.)
            M_out = torch.stack([
                s * base[:, 0],
                s * dir_flip * base[:, 1],
                s * base[:, 2],
            ], dim=1)

            tau = tau + torch.cat([F_prop_b, M_out], dim=1)
        return tau

    # ------------------------------------------------------------------
    # core batched dynamics (tensor in / tensor out)
    # ------------------------------------------------------------------
    def _dyn(self, X, U_ref):
        """dx = f(x, u) on tensors.  X:(b,15), U_ref:(b,6) -> (b,15).

        The FUSED implementation — see ``_dyn_reference`` for the readable one, which
        this must match to round-off (``tests/test_sam_golden.py``, 1e-12 relative).

        Why fused: at the planner's batch size (b ~ 1.8e5) this is memory-bandwidth
        bound, not FLOP bound, so the cost is dominated by materializing intermediate
        ``(b,6,6)`` / ``(b,3,3)`` tensors.  Working over scalar columns instead is ~3x
        faster on its own, and it is what lets ``torch.compile`` fuse the whole thing
        into a couple of kernels.  Three algebraic facts make it exact:

        1. **M is block-diagonal.**  ``MRB`` never writes its off-diagonal 3x3 blocks
           and ``MA`` is diagonal, so ``M = blkdiag(top, A)`` with ``top`` DIAGONAL and
           ``A`` a full symmetric 3x3.  (Structural, not accidental: the NumPy
           reference builds ``MRB_CO = block_diag(m_diag, J_total)``.)  Hence no
           ``torch.linalg.inv`` — a reciprocal plus a closed-form 3x3 adjugate solve.
           That also drops a host sync, since ``linalg.inv`` checks its `info` tensor.
        2. **Coriolis collapses to cross products.**  With ``M12 = 0``, ``m2c`` gives
           ``C @ nu_r = [-d1 x nr2 ; -d1 x nr1 - d2 x nr2]`` for ``d1 = top*nr1``,
           ``d2 = A @ nr2``.  No ``C`` is ever built.
        3. **The restoring force needs no transcendentals.**  ``gvect`` uses the Euler
           angles only through ``sin(th)``, ``cos(th)sin(phi)``, ``cos(th)cos(phi)`` —
           which for a DCM are exactly ``-R20``, ``R21``, ``R22`` (because
           ``cos(th) = sqrt(1 - R20^2) = sqrt(R21^2 + R22^2) >= 0`` over asin's range).
           So ``asin``, both ``atan2``s and four ``sin``/``cos`` disappear.  Physically:
           gravity's direction in the body frame is just the third row of the DCM.

        ``psi`` survives only for the water current, so the whole current block is
        skipped at trace time when ``V_c == 0`` (the default).
        """
        xv = torch.clamp(X[:, 13], 0.0, 100.0)
        xl = torch.clamp(X[:, 14], 0.0, 100.0)
        uv = torch.clamp(U_ref[:, 0], 0.0, 100.0)
        ul = torch.clamp(U_ref[:, 1], 0.0, 100.0)

        # --- attitude: normalized quaternion -> the DCM entries we actually use ---
        q0, q1, q2, q3 = X[:, 3], X[:, 4], X[:, 5], X[:, 6]
        inv_q = torch.rsqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        q0, q1, q2, q3 = q0 * inv_q, q1 * inv_q, q2 * inv_q, q3 * inv_q

        r00 = 1 - 2 * (q2 * q2 + q3 * q3)
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 1 - 2 * (q1 * q1 + q3 * q3)
        r12 = 2 * (q2 * q3 - q0 * q1)
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 1 - 2 * (q1 * q1 + q2 * q2)

        # gvect's three angle combinations, straight off the DCM (see fact 3 above).
        sth, cth_sphi, cth_cphi = -r20, r21, r22

        # --- velocities -------------------------------------------------------
        vu, vv, vw = X[:, 7], X[:, 8], X[:, 9]
        wp, wq, wr = X[:, 10], X[:, 11], X[:, 12]
        U_speed = torch.sqrt(vu * vu + vv * vv + vw * vw)

        nr1u, nr1v, nr1w = vu, vv, vw
        if self.V_c != 0.0:
            # Only now do we need the heading; skipped entirely at V_c = 0.
            psi = torch.atan2(r10, r00)
            nr1u = vu - self.V_c * torch.cos(self.beta_c - psi)
            nr1v = vv - self.V_c * torch.sin(self.beta_c - psi)

        # --- mass, CoG, inertia ------------------------------------------------
        x_vbs = self._c_xvbs * xv
        m_vbs = self._c_mvbs * xv
        m = self.m_ss + m_vbs + self.m_lcg
        inv_m = 1.0 / m

        plx = self._pLcg0[0] + self._c_xlcg * xl
        ply = self._pLcg0[1]
        plz = self._pLcg0[2]

        xg = inv_m * (self.m_ss * self._pSsg[0] + m_vbs * self._pVbs[0] + self.m_lcg * plx)
        yg = inv_m * (self.m_ss * self._pSsg[1] + m_vbs * self._pVbs[1] + self.m_lcg * ply)
        zg = inv_m * (self.m_ss * self._pSsg[2] + m_vbs * self._pVbs[2] + self.m_lcg * plz)

        Ix_vbs = 0.5 * m_vbs * self.r_vbs ** 2
        Iy_vbs = (1.0 / 12.0) * m_vbs * (3.0 * self.r_vbs ** 2 + x_vbs * x_vbs)

        # J_total = (J_ss_co + J_lcg_cg) + [diag(Ix,Iy,Iy) - m_vbs*S2_vbs] - m_lcg*S2_lcg,
        # with S2 = skew(p)@skew(p) = p p^T - |p|^2 I (symmetric).
        pl2 = plx * plx + ply * ply + plz * plz
        j00 = self._Jc00 + Ix_vbs - m_vbs * self._S2v00 - self.m_lcg * (plx * plx - pl2)
        j01 = self._Jc01 - m_vbs * self._S2v01 - self.m_lcg * (plx * ply)
        j02 = self._Jc02 - m_vbs * self._S2v02 - self.m_lcg * (plx * plz)
        j11 = self._Jc11 + Iy_vbs - m_vbs * self._S2v11 - self.m_lcg * (ply * ply - pl2)
        j12 = self._Jc12 - m_vbs * self._S2v12 - self.m_lcg * (ply * plz)
        j22 = self._Jc22 + Iy_vbs - m_vbs * self._S2v22 - self.m_lcg * (plz * plz - pl2)
        j00 = j00 * self.inertia_factor

        # A = J_total + MA_rot.  NOTE MA[5] uses J11, NOT J22 — mirroring the reference.
        a00 = j00 + self.r44 * j00
        a11 = j11 + self.k_prime * j11
        a22 = j22 + self.k_prime * j11
        a01, a02, a12 = j01, j02, j12
        t1 = m + m * self.k1
        t2 = m + m * self.k2
        t3 = t2

        # --- Coriolis (fact 2): two cross products, no matrix -------------------
        d1u, d1v, d1w = t1 * nr1u, t2 * nr1v, t3 * nr1w
        d2p = a00 * wp + a01 * wq + a02 * wr
        d2q = a01 * wp + a11 * wq + a12 * wr
        d2r = a02 * wp + a12 * wq + a22 * wr

        cn0 = -(d1v * wr - d1w * wq)
        cn1 = -(d1w * wp - d1u * wr)
        cn2 = -(d1u * wq - d1v * wp)
        cn3 = -(d1v * nr1w - d1w * nr1v) - (d2q * wr - d2r * wq)
        cn4 = -(d1w * nr1u - d1u * nr1w) - (d2r * wp - d2p * wr)
        cn5 = -(d1u * nr1v - d1v * nr1u) - (d2p * wq - d2q * wp)

        # --- restoring forces (gvect) ------------------------------------------
        W = m * self.g
        B = self.B
        rbx, rby, rbz = self._pOB
        WmB = W - B
        gx = WmB * sth
        gy = -WmB * cth_sphi
        gz = -WmB * cth_cphi
        g3 = -(yg * W - rby * B) * cth_cphi + (zg * W - rbz * B) * cth_sphi
        g4 = (zg * W - rbz * B) * sth + (xg * W - rbx * B) * cth_cphi
        g5 = -(xg * W - rbx * B) * cth_sphi - (yg * W - rby * B) * sth

        # --- propeller thrust ---------------------------------------------------
        tx, ty, tz, tp, tq, tr = self._prop_fused(U_speed, U_ref)

        # --- damping ------------------------------------------------------------
        if self.piml_type == "pinn":
            # Learned state-dependent D(nu, u): the one block that cannot be folded
            # away, since D is produced per row by the network.  `_damping_force`
            # returns D @ nu_r directly -- via `L @ (L^T nu_r)` for a Cholesky model
            # (fusible), or a dense bmm for the legacy full-A one.  `_dyn_reference`
            # deliberately keeps the dense path, so the fused-vs-reference parity test
            # is a free independent oracle for this fast path.
            nu_r = torch.stack([nr1u, nr1v, nr1w, wp, wq, wr], dim=1)
            nu = X[:, 7:13]
            u_ref6 = torch.cat([uv.unsqueeze(1), ul.unsqueeze(1), U_ref[:, 2:]], dim=1)
            Dnu = self._damping_force(torch.cat([nu, u_ref6], dim=1), nu_r)
            dn0, dn1, dn2, dn3, dn4, dn5 = Dnu.unbind(1)
        else:
            dn0, dn1, dn2 = self._Dlin * nr1u, self._Dlin * nr1v, self._Dlin * nr1w
            dn3, dn4, dn5 = self._Drot * wp, self._Drot * wq, self._Drot * wr

        rhs0 = tx - cn0 - dn0 - gx
        rhs1 = ty - cn1 - dn1 - gy
        rhs2 = tz - cn2 - dn2 - gz
        rhs3 = tp - cn3 - dn3 - g3
        rhs4 = tq - cn4 - dn4 - g4
        rhs5 = tr - cn5 - dn5 - g5

        # --- accelerations: block-diagonal solve (fact 1) -----------------------
        c00 = a11 * a22 - a12 * a12
        c01 = a02 * a12 - a01 * a22
        c02 = a01 * a12 - a02 * a11
        c11 = a00 * a22 - a02 * a02
        c12 = a02 * a01 - a00 * a12
        c22 = a00 * a11 - a01 * a01
        inv_det = 1.0 / (a00 * c00 + a01 * c01 + a02 * c02)

        nd0 = rhs0 / t1
        nd1 = rhs1 / t2
        nd2 = rhs2 / t3
        nd3 = (c00 * rhs3 + c01 * rhs4 + c02 * rhs5) * inv_det
        nd4 = (c01 * rhs3 + c11 * rhs4 + c12 * rhs5) * inv_det
        nd5 = (c02 * rhs3 + c12 * rhs4 + c22 * rhs5) * inv_det

        # --- actuator dynamics (constant tau_act; see the ctor) -----------------
        inv_tau = 1.0 / self.tau_act
        du_v = torch.clamp((uv - xv) * inv_tau, -self.vbs_dot_max, self.vbs_dot_max)
        du_l = torch.clamp((ul - xl) * inv_tau, -self.lcg_dot_max, self.lcg_dot_max)

        # --- kinematics ----------------------------------------------------------
        px = r00 * vu + r01 * vv + r02 * vw
        py = r10 * vu + r11 * vv + r12 * vw
        pz = r20 * vu + r21 * vv + r22 * vw

        # Baumgarte stabilization on the NORMALIZED quaternion, matching the reference
        # (which rebinds q to the normalized one before calling _eta_dynamics).  So this
        # is ~0 by construction — but it is not identically 0, and dropping it would
        # break golden parity, so it is computed the same way rather than folded out.
        qn2 = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3
        k = (self.gamma / 2.0) * (1.0 - qn2)
        qd0 = 0.5 * (-q1 * wp - q2 * wq - q3 * wr) + k * q0
        qd1 = 0.5 * (q0 * wp - q3 * wq + q2 * wr) + k * q1
        qd2 = 0.5 * (q3 * wp + q0 * wq - q1 * wr) + k * q2
        qd3 = 0.5 * (-q2 * wp + q1 * wq + q0 * wr) + k * q3

        return torch.stack([px, py, pz, qd0, qd1, qd2, qd3,
                            nd0, nd1, nd2, nd3, nd4, nd5, du_v, du_l], dim=1)

    def _prop_fused(self, U_speed, u_ref):
        """Propeller wrench as 6 scalar columns.  Mirrors ``_propeller_force``.

        Only column 0 of ``C_T2C = Rz(dr) @ Ry(ds)`` and the single product
        ``C_T2C @ r_sh`` are ever consumed, and both are closed-form, so the two
        ``(b,3,3)`` rotation stacks and the ``bmm`` all disappear.
        """
        ds = -u_ref[:, 2]
        dr = -u_ref[:, 3]
        cds, sds = torch.cos(ds), torch.sin(ds)
        cdr, sdr = torch.cos(dr), torch.sin(dr)
        Va_over_D = self.Va_coef * U_speed / self.D_prop

        # C_T2C column 0 = Rz @ (cds, 0, sds).
        e0 = cdr * cds
        e1 = -sdr * cds
        e2 = sds

        cx, cy, cz = self._pOC
        tx = ty = tz = tp = tq = tr = None
        for i in range(2):
            n = u_ref[:, 4 + i] / 60.0
            abs_n = torch.abs(n)
            pos = n > 0
            nn = abs_n * n

            X_prop = torch.where(
                pos,
                self._cT * (self.KT_0 * nn + self._ktsl * Va_over_D * abs_n),
                self._cT * self.KT_0 * nn / 10.0)
            K_prop = torch.where(
                pos,
                self._cQ * (self.KQ_0 * nn + self._kqsl * Va_over_D * abs_n),
                self._cQ * self.KQ_0 * nn / 10.0)
            dir_flip = torch.where(pos, torch.ones_like(n), -torch.ones_like(n))

            fx, fy, fz = X_prop * e0, X_prop * e1, X_prop * e2

            # r_prop = C_T2C @ r_sh - p_OC.  r_sh = (rx, 0, 0) for both propellers, so
            # C_T2C @ r_sh is rx * column 0 — the same column already computed above.
            rx = self._rsh[i][0]
            r0 = rx * e0 - cx
            r1 = rx * e1 - cy
            r2 = rx * e2 - cz

            b0 = r1 * fz - r2 * fy
            b1 = r2 * fx - r0 * fz
            b2 = r0 * fy - r1 * fx
            b0 = b0 + (1.0 if i % 2 == 0 else -1.0) * K_prop

            s = self.thruster_rot_strength
            if i == 0:
                tx, ty, tz = fx, fy, fz
                tp, tq, tr = s * b0, s * dir_flip * b1, s * b2
            else:
                tx, ty, tz = tx + fx, ty + fy, tz + fz
                tp = tp + s * b0
                tq = tq + s * dir_flip * b1
                tr = tr + s * b2
        return tx, ty, tz, tp, tq, tr

    def _dyn_reference(self, X, U_ref):
        """The readable, matrix-based dx = f(x, u).  Kept as the parity oracle for
        ``_dyn``; not used on the hot path.  X:(b,15), U_ref:(b,6) -> (b,15).

        State layout: eta(7) | nu(6) | act(2) where act = [x_vbs, x_lcg].  Only
        vbs/lcg are carried as states (they drive buoyancy mass / CoG / inertia);
        the fin/rpm actuation enters through the command ``U_ref`` directly.
        """
        u = self._bound_actuators(X[:, 13:15])          # (b,2) [vbs, lcg] state
        u_ref = self._bound_actuators(U_ref)            # (b,6) command

        eta = X[:, 0:7]
        nu = X[:, 7:13]

        # --- system state -------------------------------------------------
        q = eta[:, 3:7]
        q = q / torch.linalg.norm(q, dim=1, keepdim=True)
        R = self._quat_to_dcm(q)
        psi, theta, phi = self._angles_from_dcm(R)

        u_c = self.V_c * torch.cos(self.beta_c - psi)
        v_c = self.V_c * torch.sin(self.beta_c - psi)
        nu_c = torch.zeros_like(nu)
        nu_c[:, 0] = u_c
        nu_c[:, 1] = v_c
        nu_r = nu - nu_c
        U_speed = torch.linalg.norm(nu[:, 0:3], dim=1)

        x_vbs = (u[:, 0] / 100.0) * self.l_vbs_l
        m_vbs = self.rho_w * self._pi * self.r_vbs ** 2 * x_vbs
        m = self.m_ss + m_vbs + self.m_lcg

        lcg_off = torch.zeros(X.shape[0], 3, dtype=self.dtype, device=self.device)
        lcg_off[:, 0] = (u[:, 1] / 100.0) * self.l_lcg_l
        p_OLcg_O = self.p_OLcgPos_O.unsqueeze(0) + lcg_off            # (b,3)

        # --- center of gravity -------------------------------------------
        inv_m = (1.0 / m).unsqueeze(1)
        p_OG_O = (self.m_ss * inv_m) * self.p_OSsg_O.unsqueeze(0) \
            + (m_vbs.unsqueeze(1) * inv_m) * self.p_OVbs_O.unsqueeze(0) \
            + (self.m_lcg * inv_m) * p_OLcg_O

        # --- inertias -----------------------------------------------------
        Ix_vbs = 0.5 * m_vbs * self.r_vbs ** 2
        Iy_vbs = (1.0 / 12.0) * m_vbs * (3.0 * self.r_vbs ** 2 + x_vbs ** 2)
        J_vbs_cg = torch.diag_embed(torch.stack([Ix_vbs, Iy_vbs, Iy_vbs], dim=1))
        J_vbs_co = J_vbs_cg - m_vbs.view(-1, 1, 1) * self.S2_vbs.unsqueeze(0)

        S2_lcg = torch.bmm(self._skew(p_OLcg_O), self._skew(p_OLcg_O))
        J_lcg_co = self.J_lcg_cg.unsqueeze(0) - self.m_lcg * S2_lcg

        J_total = self.J_ss_co.unsqueeze(0) + J_vbs_co + J_lcg_co
        J_total = J_total.clone()
        J_total[:, 0, 0] = J_total[:, 0, 0] * self.inertia_factor

        # --- mass matrix --------------------------------------------------
        b = X.shape[0]
        MRB = torch.zeros(b, 6, 6, dtype=self.dtype, device=self.device)
        idx = torch.arange(3, device=self.device)
        MRB[:, idx, idx] = m.unsqueeze(1)
        MRB[:, 3:6, 3:6] = J_total

        MA_diag = torch.stack([
            m * self.k1,
            m * self.k2,
            m * self.k2,
            self.r44 * J_total[:, 0, 0],
            self.k_prime * J_total[:, 1, 1],
            self.k_prime * J_total[:, 1, 1],
        ], dim=1)
        MA = torch.diag_embed(MA_diag)

        M = MRB + MA
        Minv = torch.linalg.inv(M)

        # --- Coriolis (constant D handled below) --------------------------
        C = self._m2c(MRB, nu_r) + self._m2c(MA, nu_r)

        # --- restoring forces (gvect) -------------------------------------
        W = m * self.g
        g_vec = self._gvect(W, theta, phi, p_OG_O)

        # --- propeller thrust ---------------------------------------------
        tau = self._propeller_force(U_speed, u_ref)

        # --- accelerations ------------------------------------------------
        Cnu = torch.bmm(C, nu_r.unsqueeze(-1)).squeeze(-1)
        if self.piml_type == "pinn":
            # Learned D(nu, u): feature is the body velocity + the 6-D actuator
            # COMMAND u_ref (matches SAM_PIML's 12-D feature).  We feed the command
            # rather than the actuator state: the state no longer carries fins/rpm,
            # and for the fast-tracking actuators command ~= position, so D stays
            # in-distribution.  Applied to the current-relative velocity nu_r.
            D_batch = self.piml_model(torch.cat([nu, u_ref], dim=1))   # (b,6,6)
            Dnu = torch.bmm(D_batch, nu_r.unsqueeze(-1)).squeeze(-1)
        else:
            Dnu = torch.einsum("ij,bj->bi", self.D, nu_r)
        rhs = tau - Cnu - Dnu - g_vec
        nu_dot = torch.bmm(Minv, rhs.unsqueeze(-1)).squeeze(-1)

        # --- actuator dynamics (vbs/lcg only) -----------------------------
        # The 2 actuator states track their commands (u_ref[:, :2]) at the
        # rate-limited slew; fins/rpm have no state, they act through u_ref.
        # `tau_act` is a CONSTANT (not the integration substep), so this is a genuine
        # f(x, u) -- see the ctor.  Equal to the substep at the nominal step size.
        u_dot = (u_ref[:, :2] - u) / self.tau_act       # (b,2)
        u_dot = u_dot.clone()
        u_dot[:, 0] = torch.clamp(u_dot[:, 0], -self.vbs_dot_max, self.vbs_dot_max)
        u_dot[:, 1] = torch.clamp(u_dot[:, 1], -self.lcg_dot_max, self.lcg_dot_max)

        # --- kinematics ---------------------------------------------------
        eta_dot = self._eta_dynamics(R, q, nu)

        return torch.cat([eta_dot, nu_dot, u_dot], dim=1)

    def _gvect(self, W, theta, phi, p_OG_O):
        """Batched restoring-force vector, mirroring ``gnc.gvect`` (B constant)."""
        sth, cth = torch.sin(theta), torch.cos(theta)
        sphi, cphi = torch.sin(phi), torch.cos(phi)
        B = self.B
        rbb = self.p_OB_O
        xg, yg, zg = p_OG_O[:, 0], p_OG_O[:, 1], p_OG_O[:, 2]
        WmB = W - B
        g3 = -(yg * W - rbb[1] * B) * cth * cphi + (zg * W - rbb[2] * B) * cth * sphi
        g4 = (zg * W - rbb[2] * B) * sth + (xg * W - rbb[0] * B) * cth * cphi
        g5 = -(xg * W - rbb[0] * B) * cth * sphi - (yg * W - rbb[1] * B) * sth
        return torch.stack([
            WmB * sth,
            -WmB * cth * sphi,
            -WmB * cth * cphi,
            g3, g4, g5,
        ], dim=1)

    def _eta_dynamics(self, R, q, nu):
        """Batched position + quaternion kinematics, mirroring ``SAM.eta_dynamics``.

        R:(b,3,3) DCM of the (already normalised) quaternion q:(b,4); nu:(b,6).
        """
        pos_dot = torch.bmm(R, nu[:, 0:3].unsqueeze(-1)).squeeze(-1)

        om = nu[:, 3:6]
        q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        T = torch.stack([
            torch.stack([-q1, -q2, -q3], dim=1),
            torch.stack([q0, -q3, q2], dim=1),
            torch.stack([q3, q0, -q1], dim=1),
            torch.stack([-q2, q1, q0], dim=1),
        ], dim=1) * 0.5
        q_dot = torch.bmm(T, om.unsqueeze(-1)).squeeze(-1)
        qnorm2 = torch.sum(q * q, dim=1, keepdim=True)
        q_dot = q_dot + (self.gamma / 2.0) * (1.0 - qnorm2) * q
        return torch.cat([pos_dot, q_dot], dim=1)

    # ------------------------------------------------------------------
    # public API (numpy in / numpy out)
    # ------------------------------------------------------------------
    def dynamics(self, X, U_ref):
        """Batched ``dx = f(x, u)``.

        Parameters
        ----------
        X     : (b, 15) states  [eta(7) | nu(6) | act(2)=[vbs, lcg]]
        U_ref : (b, 6)  controls [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]

        Returns
        -------
        dX : (b, 15) numpy array of time derivatives.
        """
        Xt = self._t(X)
        Ut = self._t(U_ref)
        with torch.no_grad():
            dX = self._dyn(Xt, Ut)
        return dX.cpu().numpy()

    # ------------------------------------------------------------------
    # one integrator substep — the unit that gets compiled
    # ------------------------------------------------------------------
    # `torch.compile` is applied to ONE substep, never to the substep loop: compiling
    # the loop would inline `_dyn` 50-200x and take minutes.  The loop bound is a plain
    # Python int and so never enters a graph, which is what makes this decomposition work.
    #
    # `keep` is the per-row "this row is still integrating" mask for the variable-duration
    # rollout.  Zeroing `h` alone is NOT enough to freeze a row: the quaternion renorm is
    # not multiplied by `h`, so a finished row would keep drifting by ~1 ulp per extra
    # substep and its answer would depend on how many OTHER rows in its batch are still
    # running.  The final `torch.where` makes frozen rows bit-exact, which is what
    # `benchmarking/polish/harness.py::check_integrator_parity` relies on (it compares one
    # row at b=176400 against the same row at b=2, at 1e-9).

    def _step_euler(self, X, U6, h, keep):
        """One forward-Euler substep.  h:(b,1) already masked, keep:(b,1) bool."""
        Xn = self._normalize_quat(X + h * self._dyn(X, U6))
        return torch.where(keep, Xn, X)

    def _step_rk4(self, X, U6, h, keep):
        """One RK4 substep, control held constant over the substep (as the reference)."""
        k1 = self._dyn(X, U6)
        k2 = self._dyn(X + 0.5 * h * k1, U6)
        k3 = self._dyn(X + 0.5 * h * k2, U6)
        k4 = self._dyn(X + h * k3, U6)
        Xn = self._normalize_quat(X + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        return torch.where(keep, Xn, X)

    def rollout(self, X0, u_fn, h, n_steps=None, n_max=None, integrator="rk4"):
        """Batched rollout with PER-ROW step size and PER-ROW substep count.

        Tensor in / tensor out — no host round-trip, so callers can keep the whole
        trajectory on device.

        Parameters
        ----------
        X0       : (b, 15) tensor   initial states
        u_fn     : callable(tau) -> (b, 6) tensor
                   The expanded ``u_ref`` at normalized segment time ``tau`` (b,1).
                   A callable rather than a precomputed ``(b, n_max, 6)`` trace on
                   purpose: at planner scale that trace is ~0.5 GB and dominated the
                   whole rollout (measured 0.69 s to build + 0.26 s to upload).
        h        : (b, 1) tensor    per-row substep size
        n_steps  : (b, 1) int tensor, or None for "every row runs n_max steps"
        n_max    : int              loop bound; defaults to ``int(n_steps.max())``
        integrator : "euler" | "rk4"

        Returns
        -------
        X : (b, 15) tensor of forward-propagated states.  The 2 actuator dims
            (13:15) = [vbs, lcg] are proper states: they integrate (rate-limited)
            toward the command and are NOT reset across the rollout.
        """
        if integrator not in ("rk4", "euler"):
            raise ValueError(f"Unknown integrator {integrator!r} (expected 'euler' or 'rk4')")
        step = self._step_rk4_c if integrator == "rk4" else self._step_euler_c
        if n_max is None:
            if n_steps is None:
                raise ValueError("rollout needs either n_steps or an explicit n_max")
            n_max = int(n_steps.max())

        X = self._t(X0)
        with torch.no_grad():
            X = self._normalize_quat(X)
            for s in range(n_max):
                if n_steps is None:
                    # Uniform count: no mask needed, and `tau` is a plain scalar.
                    tau = torch.full_like(h, s / n_max)
                    keep = torch.ones_like(h, dtype=torch.bool)
                    h_eff = h
                else:
                    # Each row's control runs over ITS OWN horizon; a shared t-grid
                    # would stretch the controls of the short rows.
                    tau = torch.clamp(s / n_steps, max=1.0)
                    keep = s < n_steps
                    h_eff = h * keep
                X = step(X, u_fn(tau), h_eff, keep)
        return X

    # ------------------------------------------------------------------
    # numpy-facing rollouts (the API the parity tests use)
    # ------------------------------------------------------------------
    def euler_rollout(self, X0, U_ref_traces, duration, n_euler):
        """Batched coarse forward-Euler rollout over a precomputed control trace.

        Parameters
        ----------
        X0           : (b, 15)            initial states
        U_ref_traces : (b, n_euler, 6)    per-substep expanded controls (u_ref)
        duration     : float or (b,)      integration horizon (scalar or per-row)
        n_euler      : int                number of Euler substeps

        Returns
        -------
        X : (b, 15) numpy array of forward-propagated states.
        """
        return self._trace_rollout(X0, U_ref_traces, duration, n_euler, "euler")

    def rk4_rollout(self, X0, U_ref_traces, duration, n_rk4):
        """Batched RK4 rollout over a precomputed control trace.  See ``euler_rollout``."""
        return self._trace_rollout(X0, U_ref_traces, duration, n_rk4, "rk4")

    def _trace_rollout(self, X0, U_ref_traces, duration, n_sub, integrator):
        """Shared body of ``euler_rollout`` / ``rk4_rollout``: uniform substep count,
        controls read out of a precomputed ``(b, n_sub, 6)`` trace."""
        if integrator not in ("rk4", "euler"):
            raise ValueError(f"Unknown integrator {integrator!r} (expected 'euler' or 'rk4')")
        step = self._step_rk4_c if integrator == "rk4" else self._step_euler_c
        # h: scalar duration -> (1,1); per-row array (b,) -> (b,1).  Broadcasts over
        # (b, NX) in the update either way.
        h = (self._t(duration) / n_sub).reshape(-1, 1)
        U = self._t(U_ref_traces)
        keep = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            X = self._normalize_quat(self._t(X0))
            for s in range(n_sub):
                X = step(X, U[:, s, :], h, keep)
        return X.cpu().numpy()

    def _normalize_quat(self, X):
        """Renormalise X[:,3:7]; reset a degenerate quaternion to identity.  In-batch.

        Branch-free on purpose.  This runs once per substep (50-200x per rollout), and
        the previous ``if (~good).any():`` + boolean-mask assignment forced a
        device->host sync and a data-dependent output shape every single time — a hard
        blocker for ``torch.compile`` (guaranteed graph break) and for CUDA graphs.
        ``torch.where`` gives bit-identical results with no sync: where the norm is
        degenerate we scale by 0 (nulling the quaternion) and add 1 to the scalar part,
        which is exactly the identity quaternion.
        """
        q = X[:, 3:7]
        nrm = torch.linalg.norm(q, dim=1, keepdim=True)
        good = nrm > 1e-9
        inv = torch.where(good, 1.0 / nrm, torch.zeros_like(nrm))
        q_safe = q * inv
        # Degenerate rows are all-zero here; setting q0 = 1 makes them the identity.
        q_safe = torch.cat([q_safe[:, :1] + (~good), q_safe[:, 1:]], dim=1)
        return torch.cat([X[:, :3], q_safe, X[:, 7:]], dim=1)
