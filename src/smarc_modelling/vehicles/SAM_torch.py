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

from time import time
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
                 device="auto", dtype=torch.float64, piml_type=None):
        # Reference NumPy model: single source of truth for every constant.
        ref = _SAMNumpy(dt=dt, V_current=V_current, beta_current=beta_current)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.dt = float(dt)

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
            from smarc_modelling.piml.pinn.pinn import load_pinn_D
            self.piml_model = load_pinn_D().to(device=self.device, dtype=self.dtype)

        self._pi = math.pi

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _t(self, arr):
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

        State layout: eta(7) | nu(6) | act(2) where act = [x_vbs, x_lcg].  Only
        vbs/lcg are carried as states (they drive buoyancy mass / CoG / inertia);
        the fin/rpm actuation enters through the command ``U_ref`` directly.
        """
        u = self._bound_actuators(X[:, 13:15])          # (b,2) [vbs, lcg] state
        u_ref = self._bound_actuators(U_ref)            # (b,6) command

        eta = X[:, 0:7]
        nu = X[:, 7:13]

        # create a time counter to keep track of how much time spent on inverses
        t_inverses = 0.0

        # --- system state -------------------------------------------------
        q = eta[:, 3:7]
        t0 = time()
        q = q / torch.linalg.norm(q, dim=1, keepdim=True)
        t_inverses += time() - t0
        R = self._quat_to_dcm(q)
        psi, theta, phi = self._angles_from_dcm(R)

        u_c = self.V_c * torch.cos(self.beta_c - psi)
        v_c = self.V_c * torch.sin(self.beta_c - psi)
        nu_c = torch.zeros_like(nu)
        nu_c[:, 0] = u_c
        nu_c[:, 1] = v_c
        nu_r = nu - nu_c
        t0 = time()
        U_speed = torch.linalg.norm(nu[:, 0:3], dim=1)
        t_inverses += time() - t0

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
        t0 = time()
        Minv = torch.linalg.inv(M)
        t_inverses += time() - t0
        # print(f"Time spent on inverses: {t_inverses:.6f} s")

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
        u_dot = (u_ref[:, :2] - u) / self.dt            # (b,2)
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

    def euler_rollout(self, X0, U_ref_traces, duration, n_euler):
        """Batched coarse forward-Euler rollout, mirroring ``sam.py._euler_rollout``.

        Parameters
        ----------
        X0           : (b, 15)            initial states
        U_ref_traces : (b, n_euler, 6)    per-substep expanded controls (u_ref)
        duration     : float or (b,)      integration horizon (scalar or per-row)
        n_euler      : int                number of Euler substeps

        Returns
        -------
        X : (b, 15) numpy array of forward-propagated states.  The 2 actuator dims
            (13:15) = [vbs, lcg] are proper states: they integrate (rate-limited)
            toward the command and are NOT reset across the rollout.
        """
        # h: scalar duration -> (1,1); per-row array (b,) -> (b,1).  Broadcasts over
        # (b, NX) in the Euler update and over the actuator term inside _dyn either way.
        h = (self._t(duration) / n_euler).reshape(-1, 1)
        self.dt = h                          # drive actuator dynamics on the substep
        X = self._t(X0)
        U = self._t(U_ref_traces)
        with torch.no_grad():
            X = self._normalize_quat(X)
            for s in range(n_euler):
                X = X + h * self._dyn(X, U[:, s, :])
                X = self._normalize_quat(X)
        return X.cpu().numpy()
    
    def rk4_rollout(self, X0, U_ref_traces, duration, n_rk4):
        """Batched RK4 rollout, mirroring ``sam.py._rk4_rollout``.

        Parameters
        ----------
        X0           : (b, 15)            initial states
        U_ref_traces : (b, n_rk4, 6)      per-substep expanded controls (u_ref)
        duration     : float or (b,)      integration horizon (scalar or per-row)
        n_rk4       : int                 number of RK4 substeps

        Returns
        -------
        X : (b, 15) numpy array of forward-propagated states.  The 2 actuator dims
            (13:15) = [vbs, lcg] are proper states: they integrate (rate-limited)
            toward the command and are NOT reset across the rollout.
        """
        # h: scalar duration -> (1,1); per-row array (b,) -> (b,1).  Broadcasts over
        # (b, NX) in the RK4 stages and over the actuator term inside _dyn either way.
        h = (self._t(duration) / n_rk4).reshape(-1, 1)
        self.dt = h                          # drive actuator dynamics on the substep
        X = self._t(X0)
        U = self._t(U_ref_traces)
        with torch.no_grad():
            X = self._normalize_quat(X)
            for s in range(n_rk4):
                k1 = self._dyn(X, U[:, s, :])
                k2 = self._dyn(X + 0.5 * h * k1, U[:, s, :])
                k3 = self._dyn(X + 0.5 * h * k2, U[:, s, :])
                k4 = self._dyn(X + h * k3, U[:, s, :])
                X = X + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                X = self._normalize_quat(X)
        return X.cpu().numpy()

    def _normalize_quat(self, X):
        """Renormalise X[:,3:7]; reset a degenerate quaternion to identity.  In-batch."""
        X = X.clone()
        q = X[:, 3:7]
        nrm = torch.linalg.norm(q, dim=1, keepdim=True)
        good = (nrm > 1e-9).squeeze(1)
        q_safe = torch.where(nrm > 1e-9, q / nrm, q)
        X[:, 3:7] = q_safe
        if (~good).any():
            ident = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                 dtype=self.dtype, device=self.device)
            X[~good, 3:7] = ident
        return X
