#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM_casadi_bundle.py:

   CasADi (symbolic) port of the **15-D reduced** SAM dynamics that the
   Graphs-of-Bundles planner actually integrates, i.e. a mirror of
   ``SAM_torch.SAMTorch`` (which ``classes/robots/sam.py`` uses for BOTH the bundle
   rollout and the scalar rollout).

   Why not the existing ``SAM_casadi.py``?
   --------------------------------------
   That file is the vendored **19-D** CasADi model mirroring ``SAM.py``'s *intent*,
   and it differs from what the bundle actually runs in two ways:

     1. **Damping.**  ``SAM.py`` calls ``calculate_D()`` and then OVERWRITES it with
        a constant diagonal (``self.D = np.eye(6) * damping_factor``), so the
        elaborate nonlinear quadratic damping is dead code.  ``SAM_casadi.py`` calls
        ``calculate_D()`` and *uses* it — a genuinely different model.
     2. **State.**  19-D (carries fin/rpm actuator states) vs the bundle's 15-D
        ``[eta(7) | nu(6) | vbs, lcg]``.

   So a like-for-like feasibility study needs this file.  It is a MIRROR of
   ``SAM_torch.SAMTorch._dyn``; ``test_sam_casadi_parity.py`` asserts agreement to
   < 1e-9 on random states/controls.  If you change one, change the other.

   IMPORTANT — ``dt`` sets the actuator time constant ``tau_act``, and ``sam.py`` builds
   ``SAMTorch(dt=self.dt / self.n_integrator)``, so construct this class the same way or
   the vbs/lcg slew will not match.  ``tau_act`` is a CONSTANT, not the integration
   substep: the actuator dynamics ``u_dot = (u_ref - u)/tau_act`` (rate-clamped) must not
   depend on the discretization, or a bundle whose samples integrate different horizons
   would mix actuator models.  Pass ``tau_act`` explicitly to override.

   Non-smoothness (matters for IPOPT, not for torch)
   -------------------------------------------------
   The mirrored physics has three kinks, ported as ``fmin/fmax``/``if_else``:
     * ``_bound_actuators`` : clamp(vbs/lcg cmd, 0, 100)
     * actuator rate limit  : clamp(u_dot, +-*_dot_max)
     * propeller fwd/rev    : if_else(n > 0, X_pos, X_neg)   <- kink at n = 0
   ``smooth=True`` replaces the propeller branch with a tanh blend and the clamps
   with softplus surrogates (what IPOPT wants); ``smooth=False`` (default) is the
   exact torch mirror used by the parity test.

Author: added for the bundle-stl SAM infeasibility analysis.
"""
import numpy as np
import casadi as cs

from smarc_modelling.vehicles.SAM import SAM as _SAMNumpy


def _skew_np(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _skew(v):
    return cs.vertcat(cs.horzcat(0, -v[2], v[1]),
                      cs.horzcat(v[2], 0, -v[0]),
                      cs.horzcat(-v[1], v[0], 0))


def _softplus(a, eps):
    """eps*log(1 + exp(a/eps)), evaluated stably.

    The naive form overflows for a >> eps, and clamping the EXPONENT (a common
    mistake) makes it SATURATE at eps*cap instead of growing like ``a``.  The
    identity ``eps*log(1+exp(t)) = max(t,0)*eps + eps*log(1+exp(-|t|))`` is exact
    and overflow-free; the fmax/fabs kinks cancel, so the result is still smooth.
    """
    t = a / eps
    return cs.fmax(a, 0.0) + eps * cs.log(1.0 + cs.exp(-cs.fabs(t)))


def _clamp(x, lo, hi, smooth=False, eps=1e-3):
    """Exact clamp, or a smooth softplus surrogate when smooth=True."""
    if not smooth:
        return cs.fmin(cs.fmax(x, lo), hi)
    return hi - _softplus(hi - (lo + _softplus(x - lo, eps)), eps)


class SAMCasadiBundle:
    """Symbolic 15-D SAM dynamics mirroring ``SAM_torch.SAMTorch``."""

    def __init__(self, dt=0.05, V_current=0.0, beta_current=0.0, smooth=False,
                 vel_eps=0.0, tau_act=None):
        """
        vel_eps : float
            Regularises the speed magnitude as ``sqrt(|nu_xyz|^2 + vel_eps)``.

            REQUIRED (> 0) for any NLP use.  The torch model computes
            ``U_speed = norm_2(nu[0:3])``, whose derivative ``nu/|nu|`` is 0/0 = NaN at
            ZERO VELOCITY — and the SAM scenarios start (and end) AT REST, so the very
            first defect constraint's Jacobian is NaN (IPOPT:
            ``Invalid_Number_Detected``).  Torch never noticed because it only ever
            evaluates, never differentiates.  ``vel_eps=1e-8`` moves U_speed by 1e-4
            m/s at rest (a negligible advance-ratio term) and makes the derivative
            finite.  Keep 0.0 for the exact-parity test.
        """
        ref = _SAMNumpy(dt=dt, V_current=V_current, beta_current=beta_current)
        self.dt = float(dt)
        # Actuator tracking time constant, mirroring SAMTorch.tau_act.  Defaults to `dt`,
        # so this is a rename with NO numeric change -- but it decouples the actuator
        # model from the integration substep, which is what lets `rollout` be called with
        # `duration/n_sub != dt`.
        self.tau_act = float(dt if tau_act is None else tau_act)
        self.smooth = bool(smooth)
        self.vel_eps = float(vel_eps)

        self.g = float(ref.g); self.rho = float(ref.rho); self.rho_w = float(ref.rho_w)
        self.B = float(ref.B); self.gamma = float(ref.gamma)
        self.m_ss = float(ref.ss.m_ss); self.m_lcg = float(ref.lcg.m_lcg)
        self.r_vbs = float(ref.vbs.r_vbs); self.l_vbs_l = float(ref.vbs.l_vbs_l)
        self.l_lcg_l = float(ref.lcg.l_lcg_l); self.h_lcg_dim = float(ref.lcg.h_lcg_dim)
        self.k1 = float(ref.k1); self.k2 = float(ref.k2)
        self.k_prime = float(ref.k_prime); self.r44 = float(ref.r44)
        self.inertia_factor = float(ref.inertia_factor)
        self.damping_factor = float(ref.damping_factor)
        self.damping_rot = float(ref.damping_rot)
        self.thruster_rot_strength = float(ref.thruster_rot_strength)
        self.D_prop = float(ref.D_prop); self.Va_coef = float(ref.Va_coef)
        self.KT_0 = float(ref.KT_0); self.KQ_0 = float(ref.KQ_0)
        self.KT_max = float(ref.KT_max); self.KQ_max = float(ref.KQ_max)
        self.Ja_max = float(ref.Ja_max)
        self.V_c = float(ref.V_c); self.beta_c = float(ref.beta_c)
        self.vbs_dot_max = float(ref.vbs.x_vbs_dot_max)
        self.lcg_dot_max = float(ref.lcg.x_lcg_dot_max)

        self.p_OSsg_O = np.asarray(ref.ss.p_OSsg_O, float)
        self.p_OVbs_O = np.asarray(ref.vbs.p_OVbs_O, float)
        self.p_OLcgPos_O = np.asarray(ref.lcg.p_OLcgPos_O, float)
        self.p_OC_O = np.asarray(ref.p_OC_O, float)
        self.p_OB_O = np.asarray(ref.p_OB_O, float)
        self.r_t_p_sh = [np.asarray(r, float) for r in ref.propellers.r_t_p_sh]

        a, b = float(ref.a), float(ref.b)
        Ix_ss = (2.0 / 5.0) * self.m_ss * b ** 2
        Iy_ss = (1.0 / 5.0) * self.m_ss * (a ** 2 + b ** 2)
        J_ss_cg = np.diag([Ix_ss, Iy_ss, Iy_ss])
        self.J_ss_co = J_ss_cg - self.m_ss * (_skew_np(self.p_OSsg_O) @ _skew_np(self.p_OSsg_O))
        self.S2_vbs = _skew_np(self.p_OVbs_O) @ _skew_np(self.p_OVbs_O)
        Ix_lcg = 0.5 * self.m_lcg * (self.h_lcg_dim / 2.0) ** 2
        Iy_lcg = (1.0 / 12.0) * self.m_lcg * (3.0 * (self.h_lcg_dim / 2.0) ** 2
                                              + self.l_lcg_l ** 2)
        self.J_lcg_cg = np.diag([Ix_lcg, Iy_lcg, Iy_lcg])

        D = np.eye(6) * self.damping_factor      # the constant-D OVERWRITE (see docstring)
        D[3, 3] = D[4, 4] = D[5, 5] = self.damping_rot
        self.D = D

        self._build_functions()

    # ------------------------------------------------------------------
    def _quat_to_dcm(self, q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return cs.vertcat(
            cs.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            cs.horzcat(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            cs.horzcat(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))

    def _m2c(self, M, nu):
        Msym = 0.5 * (M + M.T)
        M11, M12 = Msym[0:3, 0:3], Msym[0:3, 3:6]
        M21, M22 = M12.T, Msym[3:6, 3:6]
        d1 = M11 @ nu[0:3] + M12 @ nu[3:6]
        d2 = M21 @ nu[0:3] + M22 @ nu[3:6]
        S1, S2 = _skew(d1), _skew(d2)
        return cs.vertcat(cs.horzcat(cs.SX.zeros(3, 3), -S1),
                          cs.horzcat(-S1, -S2))

    def _bound_actuators(self, v):
        head = cs.vertcat(_clamp(v[0], 0.0, 100.0, self.smooth),
                          _clamp(v[1], 0.0, 100.0, self.smooth))
        return head if v.shape[0] <= 2 else cs.vertcat(head, v[2:])

    def _propeller_force(self, U_speed, u_ref):
        delta_s, delta_r = -u_ref[2], -u_ref[3]
        Va = self.Va_coef * U_speed
        cds, sds = cs.cos(delta_s), cs.sin(delta_s)
        cdr, sdr = cs.cos(delta_r), cs.sin(delta_r)
        Ry = cs.vertcat(cs.horzcat(cds, 0, -sds), cs.horzcat(0, 1, 0),
                        cs.horzcat(sds, 0, cds))
        Rz = cs.vertcat(cs.horzcat(cdr, sdr, 0), cs.horzcat(-sdr, cdr, 0),
                        cs.horzcat(0, 0, 1))
        C_T2C = Rz @ Ry

        tau = cs.SX.zeros(6)
        for i in range(2):
            n = u_ref[4 + i] / 60.0
            abs_n = cs.fabs(n)
            X_pos = self.rho * (self.D_prop ** 4) * (
                self.KT_0 * abs_n * n
                + (self.KT_max - self.KT_0) / self.Ja_max * (Va / self.D_prop) * abs_n)
            X_neg = self.rho * (self.D_prop ** 4) * (self.KT_0 * abs_n * n) / 10.0
            K_pos = self.rho * (self.D_prop ** 5) * (
                self.KQ_0 * abs_n * n
                + (self.KQ_max - self.KQ_0) / self.Ja_max * (Va / self.D_prop) * abs_n)
            K_neg = self.rho * (self.D_prop ** 5) * self.KQ_0 * abs_n * n / 10.0

            if self.smooth:
                w = 0.5 * (1.0 + cs.tanh(n / 1e-2))
                X_prop = w * X_pos + (1 - w) * X_neg
                K_prop = w * K_pos + (1 - w) * K_neg
                dir_flip = cs.tanh(n / 1e-2)
            else:
                X_prop = cs.if_else(n > 0, X_pos, X_neg)
                K_prop = cs.if_else(n > 0, K_pos, K_neg)
                dir_flip = cs.if_else(n > 0, 1.0, -1.0)

            F_prop_b = X_prop * C_T2C[:, 0]
            r_prop = C_T2C @ cs.DM(self.r_t_p_sh[i]) - cs.DM(self.p_OC_O)
            base = cs.cross(r_prop, F_prop_b)
            sign_i = 1.0 if (i % 2 == 0) else -1.0
            s = self.thruster_rot_strength
            M_out = cs.vertcat(s * (base[0] + sign_i * K_prop),
                               s * dir_flip * base[1],
                               s * base[2])
            tau = tau + cs.vertcat(F_prop_b, M_out)
        return tau

    def _gvect(self, W, theta, phi, p_OG_O):
        sth, cth = cs.sin(theta), cs.cos(theta)
        sphi, cphi = cs.sin(phi), cs.cos(phi)
        B, rbb = self.B, self.p_OB_O
        xg, yg, zg = p_OG_O[0], p_OG_O[1], p_OG_O[2]
        WmB = W - B
        g3 = -(yg * W - rbb[1] * B) * cth * cphi + (zg * W - rbb[2] * B) * cth * sphi
        g4 = (zg * W - rbb[2] * B) * sth + (xg * W - rbb[0] * B) * cth * cphi
        g5 = -(xg * W - rbb[0] * B) * cth * sphi - (yg * W - rbb[1] * B) * sth
        return cs.vertcat(WmB * sth, -WmB * cth * sphi, -WmB * cth * cphi, g3, g4, g5)

    def _eta_dynamics(self, R, q, nu):
        pos_dot = R @ nu[0:3]
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        T = 0.5 * cs.vertcat(cs.horzcat(-q1, -q2, -q3),
                             cs.horzcat(q0, -q3, q2),
                             cs.horzcat(q3, q0, -q1),
                             cs.horzcat(-q2, q1, q0))
        q_dot = T @ nu[3:6] + (self.gamma / 2.0) * (1.0 - cs.dot(q, q)) * q
        return cs.vertcat(pos_dot, q_dot)

    # ------------------------------------------------------------------
    def _dyn_expr(self, X, U_ref):
        """dx = f(x, u_ref). Mirrors SAMTorch._dyn.  X:(15,) U_ref:(6,) -> (15,)."""
        u = self._bound_actuators(X[13:15])
        u_ref = self._bound_actuators(U_ref)

        nu = X[7:13]
        q = X[3:7] / cs.norm_2(X[3:7])
        R = self._quat_to_dcm(q)
        theta = cs.asin(_clamp(-R[2, 0], -1.0, 1.0, self.smooth))
        phi = cs.atan2(R[2, 1], R[2, 2])
        psi = cs.atan2(R[1, 0], R[0, 0])

        nu_c = cs.vertcat(self.V_c * cs.cos(self.beta_c - psi),
                          self.V_c * cs.sin(self.beta_c - psi),
                          cs.SX.zeros(4))
        nu_r = nu - nu_c
        # norm_2 has a NaN derivative at nu=0 (the at-rest boundary states) — see vel_eps.
        U_speed = (cs.norm_2(nu[0:3]) if self.vel_eps == 0.0
                   else cs.sqrt(cs.sumsqr(nu[0:3]) + self.vel_eps))

        x_vbs = (u[0] / 100.0) * self.l_vbs_l
        m_vbs = self.rho_w * np.pi * self.r_vbs ** 2 * x_vbs
        m = self.m_ss + m_vbs + self.m_lcg

        p_OLcg_O = cs.DM(self.p_OLcgPos_O) + cs.vertcat((u[1] / 100.0) * self.l_lcg_l, 0, 0)
        p_OG_O = (self.m_ss / m) * cs.DM(self.p_OSsg_O) \
            + (m_vbs / m) * cs.DM(self.p_OVbs_O) + (self.m_lcg / m) * p_OLcg_O

        Ix_vbs = 0.5 * m_vbs * self.r_vbs ** 2
        Iy_vbs = (1.0 / 12.0) * m_vbs * (3.0 * self.r_vbs ** 2 + x_vbs ** 2)
        J_vbs_co = cs.diag(cs.vertcat(Ix_vbs, Iy_vbs, Iy_vbs)) - m_vbs * cs.DM(self.S2_vbs)
        J_lcg_co = cs.DM(self.J_lcg_cg) - self.m_lcg * (_skew(p_OLcg_O) @ _skew(p_OLcg_O))
        J_total = cs.DM(self.J_ss_co) + J_vbs_co + J_lcg_co
        J_total[0, 0] = J_total[0, 0] * self.inertia_factor

        MRB = cs.SX.zeros(6, 6)
        for i in range(3):
            MRB[i, i] = m
        MRB[3:6, 3:6] = J_total
        MA = cs.diag(cs.vertcat(m * self.k1, m * self.k2, m * self.k2,
                                self.r44 * J_total[0, 0],
                                self.k_prime * J_total[1, 1],
                                self.k_prime * J_total[1, 1]))
        M = MRB + MA

        C = self._m2c(MRB, nu_r) + self._m2c(MA, nu_r)
        g_vec = self._gvect(m * self.g, theta, phi, p_OG_O)
        tau = self._propeller_force(U_speed, u_ref)
        rhs = tau - C @ nu_r - cs.DM(self.D) @ nu_r - g_vec
        nu_dot = cs.solve(M, rhs)

        u_dot_raw = (u_ref[0:2] - u) / self.tau_act   # constant, NOT the substep
        u_dot = cs.vertcat(
            _clamp(u_dot_raw[0], -self.vbs_dot_max, self.vbs_dot_max, self.smooth),
            _clamp(u_dot_raw[1], -self.lcg_dot_max, self.lcg_dot_max, self.smooth))

        return cs.vertcat(self._eta_dynamics(R, q, nu), nu_dot, u_dot)

    def _build_functions(self):
        X = cs.SX.sym("X", 15)
        U = cs.SX.sym("U", 6)
        self.f = cs.Function("f_sam", [X, U], [self._dyn_expr(X, U)],
                             ["x", "u_ref"], ["dx"])
        h = cs.SX.sym("h")
        k1 = self._dyn_expr(X, U)
        k2 = self._dyn_expr(X + 0.5 * h * k1, U)
        k3 = self._dyn_expr(X + 0.5 * h * k2, U)
        k4 = self._dyn_expr(X + h * k3, U)
        Xn = X + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.rk4_step = cs.Function("rk4_sam", [X, U, h], [Xn],
                                    ["x", "u_ref", "h"], ["x_next"])
        # One substep INCLUDING the renormalisation the torch rollout does every step.
        self.step = cs.Function("step_sam", [X, U, h], [self._normalize_quat(Xn)],
                                ["x", "u_ref", "h"], ["x_next"])
        self._seg_cache = {}

    def segment_fn(self, n_sub):
        """A Function (x, u_ref, duration) -> x after ``n_sub`` substeps, via mapaccum.

        Use this (not the Python-loop ``rollout``) inside an NLP.  Inlining n_sub RK4
        steps per segment into the MX graph makes CasADi differentiate through
        n_sub*4 copies of the 6x6-solve dynamics — for n_sub=50 over 9 segments that
        is ~1800 dynamics evaluations and the nlpsol construction never finishes.
        ``mapaccum`` chains the step as a single structured node instead, so the graph
        stays small and the derivative is built once.  Cached per n_sub.
        """
        if n_sub in self._seg_cache:
            return self._seg_cache[n_sub]
        acc = self.step.mapaccum("acc", n_sub)
        x0 = cs.MX.sym("x0", 15)
        u = cs.MX.sym("u", 6)
        dur = cs.MX.sym("dur")
        Xall = acc(self._normalize_quat(x0),          # entry normalise (torch does this)
                   cs.repmat(u, 1, n_sub),
                   cs.repmat(dur / n_sub, 1, n_sub))
        F = cs.Function(f"F_seg{n_sub}", [x0, u, dur], [Xall[:, -1]],
                        ["x", "u_ref", "duration"], ["x_next"])
        self._seg_cache[n_sub] = F
        return F

    @staticmethod
    def _normalize_quat(X):
        """Renormalise the quaternion block, mirroring ``SAMTorch._normalize_quat``.

        The torch rollout renormalises after EVERY substep (RK4 + the gamma term let
        |q| drift), so a rollout that skips this diverges from the bundle's dynamics
        in exactly the attitude/rate dims.  The degenerate-|q|~0 reset-to-identity in
        the torch version is intentionally NOT mirrored: it is an unreachable branch
        for any sane trajectory and would inject a non-smooth if_else into the NLP.
        """
        return cs.vertcat(X[0:3], X[3:7] / cs.norm_2(X[3:7]), X[7:15])

    def rollout(self, X, U_ref, duration, n_sub):
        """RK4 rollout over ``duration`` in ``n_sub`` substeps at constant U_ref.

        Mirrors ``SAMTorch.rk4_rollout`` for a control held constant on the segment,
        INCLUDING its per-substep quaternion renormalisation (and the one on entry).
        Symbolic-friendly (X/U_ref may be SX/MX/DM).

        NOTE: ``duration/n_sub`` no longer has to equal the ``dt`` this object was
        constructed with -- the actuator slew runs on the constant ``tau_act`` (torch
        does the same; it used to re-assign ``self.dt = h``
        inside its rollout; here it is fixed at construction)."""
        h = duration / n_sub
        X = self._normalize_quat(X)
        for _ in range(n_sub):
            X = self.rk4_step(X, U_ref, h)
            X = self._normalize_quat(X)
        return X

    @staticmethod
    def expand_control(c5, rpm_max, rpm_rev_max, rev_thrust_ratio):
        """5-D reduced control -> 6-D u_ref, mirroring ``SAM._expand_control_batch``.

        c5 = [x_vbs, x_lcg, delta_s, delta_r, throttle s in [-1,1]].  Bipolar
        thrust-affine map: s=+1 -> +rpm_max, s=-1 -> -rpm_rev_max, thrust affine in s.

        WARNING — do NOT put this inside an NLP.  ``rpm = sign(P)*sqrt(|P|)`` has an
        INFINITE derivative at the coast point P=0 (d rpm/dP = 1/(2 sqrt(P))), so the
        chain rule hands IPOPT a 0*inf = NaN there (observed as
        ``Invalid_Number_Detected``).  The thrust is affine in ``s`` and the physics
        is smooth in ``rpm``; only this intermediate reparametrisation is singular.
        For optimisation, use ``u_ref_from_rpm`` and treat SIGNED RPM as the decision
        variable — the s<->rpm map is a monotone bijection, so feasibility is identical
        and ``throttle_of_rpm`` converts back for reporting.
        """
        s = c5[4]
        P_fwd = rpm_max ** 2
        P_rev = -(rpm_rev_max ** 2) / rev_thrust_ratio
        P = P_rev + (s + 1.0) / 2.0 * (P_fwd - P_rev)
        rpm = cs.if_else(P >= 0, cs.sqrt(cs.fmax(P, 1e-16)),
                         -cs.sqrt(cs.fmax(-rev_thrust_ratio * P, 1e-16)))
        return cs.vertcat(c5[0], c5[1], c5[2], c5[3], rpm, rpm)

    @staticmethod
    def u_ref_from_rpm(c5):
        """[vbs, lcg, delta_s, delta_r, SIGNED rpm] -> 6-D u_ref.

        The NLP-friendly control parametrisation (see ``expand_control``): the prop
        thrust ``~ |n| n`` is C1 in rpm, so this has no singular derivative.
        """
        return cs.vertcat(c5[0], c5[1], c5[2], c5[3], c5[4], c5[4])

    @staticmethod
    def throttle_of_rpm(rpm, rpm_max, rpm_rev_max, rev_thrust_ratio):
        """Invert the bipolar map: signed rpm -> throttle s in [-1, 1] (numpy scalar)."""
        P = (rpm ** 2) if rpm >= 0 else (-(rpm ** 2) / rev_thrust_ratio)
        P_fwd = rpm_max ** 2
        P_rev = -(rpm_rev_max ** 2) / rev_thrust_ratio
        return 2.0 * (P - P_rev) / (P_fwd - P_rev) - 1.0
