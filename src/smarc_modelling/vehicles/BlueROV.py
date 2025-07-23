#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlueROV.py:

   Class for the BlueROV 

   Actuator systems:
    8 thrusters in the heavy configuration to allow 6DoF motions.

   Sensor systems:
   - **IMU**: Inertial Measurement Unit for attitude and acceleration.
   - **DVL**: Doppler Velocity Logger for measuring underwater velocity.
   - **GPS**: For surface position tracking.
   - **Sonar**: For environment sensing during navigation and inspections.

   BlueROV()
       Step input for force and torque control input

Methods:

    [xdot] = dynamics(x, u_ref) returns for integration

    u_ref: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]


References:

    Bhat, S., Panteli, C., Stenius, I., & Dimarogonas, D. V. (2023). Nonlinear model predictive control for hydrobatic AUVs:
        Experiments with the SAM vehicle. Journal of Field Robotics, 40(7), 1840-1859. doi:10.1002/rob.22218.

    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion Control. 2nd Edition, Wiley.
        URL: www.fossen.biz/wiley

Author:     David Doerner
"""

import numpy as np
import math
import casadi as cs
from scipy.linalg import block_diag
from Utilities.smarc_modelling.src.smarc_modelling.lib.gnc import *
from Utilities.sets import HyperRectangle
from Utilities.smarc_modelling.src.smarc_modelling.lib.gnc_casadi import *

class SolidStructure:
    """
    Represents the Solid Structure (SS) of the SAM AUV.

    Attributes:
        l_SS: Length of the solid structure (m).
        d_SS: Diameter of the solid structure (m).
        m_SS: Mass of the solid structure (kg).
        p_CSsg_O: Vector from frame C to CG of SS expressed in O (m)
        p_OSsg_O: Vector from CO to CG of SS expressed in O (m)
    """

    def __init__(self, l_ss, d_ss, m_ss, p_CSsg_O, p_OC_O):
        self.l_ss = l_ss
        self.d_ss = d_ss
        self.m_ss = m_ss
        self.p_CSsg_O = p_CSsg_O
        self.p_OSsg_O = p_OC_O + self.p_CSsg_O



# Class Vehicle
class BlueROV():
    """
    BlueROV()
        Integrates all subsystems of the BlueROV.

    Attributes:
        eta: [x, y, z, q0, q1, q2, q3] - Position and quaternion orientation
        nu: [u, v, w, p, q, r] - Body-fixed linear and angular velocities

    Vectors follow Tedrake's monogram:
    https://manipulation.csail.mit.edu/pick.html#monogram
    """
    def __init__(
            self,
            dt=0.02,
            V_current=0,
            beta_current=0,
            iX=cs.SX
    ):
        self.dt = dt # Sim time step, necessary for evaluation of the actuator dynamics
        self.iX = iX

        # Constants
        self.p_OC_O = self.iX([0., 0., 0.])  # Measurement frame C in CO (O)
        self.D2R = cs.sqrt(math.pi / 180)  # Degrees to radians
        self.rho_w = self.rho = 1026  # Water density (kg/m³)
        self.g = 9.81  # Gravity acceleration (m/s²)

        # Initialize Subsystems:
        self.init_vehicle()

        # Reference values and current
        self.V_c = V_current  # Current water speed
        self.beta_c = beta_current * self.D2R  # Current water direction (rad)

        # Initialize state vectors
        self.nu = self.iX.zeros(6,1)  # [u, v, w, p, q, r]
        self.eta = self.iX.zeros(7,1)  # [x, y, z, q0, q1, q2, q3]
        self.eta[3] = 1.0

        # Initialize the AUV model
        self.name = ("BlueROV")

        # Rigid-body mass matrself.iX expressed in CO
        self.m = self.ss.m_ss
        self.p_OG_O = self.iX([0., 0., 0.])  # CG w.r.t. to the CO
        self.p_OB_O = self.iX([0., 0., 0.])  # CB w.r.t. to the CO

        # Weight and buoyancy
        self.W = self.m * self.g
        self.B = self.W 

        # Inertias from von Benzon 2022
        self.Ix = 0.26
        self.Iy = 0.23
        self.Iz = 0.37

        self.Xu = 13.7
        self.Yv = 0
        self.Zw = 33.0
        self.Kp = 0
        self.Mq = 0.8
        self.Nr = 0

        # Added mass terms
        self.Xdu = 6.36
        self.Ydv = 7.12
        self.Zdw = 18.68
        self.Kdp = 0.189
        self.Mdq = 0.135
        self.Ndr = 0.222

        # Damping coefficients
        self.Xuu = 141.0     # x-damping
        self.Yvv = 217.0 # y-damping
        self.Zww = 190.0# z-damping
        self.Kpp = 1.19 # Roll damping
        self.Mqq = 0.47 # Pitch damping
        self.Nrr = 1.5 # Yaw damping

        # System matrices
        self.MRB = cs.diag(self.iX([self.m, self.m, self.m, self.Ix, self.Iy, self.Iz]))
        self.MA = cs.diag(self.iX([self.Xdu, self.Ydv, self.Zdw, self.Kdp, self.Mdq, self.Ndr]))
        self.M = self.MRB + self.MA
        self.Minv = cs.inv(self.M)

        self.C = self.iX.zeros((6,6))

        self.D = self.iX.zeros((6,6))
        self.D_lin = cs.diag(self.iX([self.Xu, self.Yv, self.Zw, self.Kp, self.Mq, self.Nr]))
        self.D_nl = self.iX.zeros((6,6))

        self.gamma = 100 # Scaling factor for numerical stability of quaternion differentiation
        
        self.x_prev = None  # Old state vector for numerical integration
        self.U = HyperRectangle(np.array([-85, -85, -120, -26, -14, -22]),
                                np.array([85, 85, 120, 26, 14, 22]))
        
        self.create_dynamics()
        self.create_C()
        self.create_D()
        self.create_g()
        self.create_eta_dynamics()
        self.create_system_state()
        self.create_fx()
        self.create_gx()

    def init_vehicle(self):
        """
        Initialize all subsystems based on their respective parameters
        """
        self.ss = SolidStructure(
            l_ss=0.46,
            d_ss=0.58,
            m_ss=13.5,
            p_CSsg_O = self.iX([0., 0, 0.]),
            p_OC_O=self.p_OC_O
        )

    def step(self, x, u, dt):
        """
        Naive step integration with a custom time step
        Args:
            x: state space vector with [eta, nu]
            u: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
            dt: time step for integration

        Returns:
            x_next: state space vector at next time step
        """
        if self.x_prev is None:
            self.x_prev = x
        # Calculate the dynamics
        x_dot = self.calculate_fx(x) + self.calculate_gx(x)@u #self.dynamics(x, u)
        # Update the state using Euler integration
        x_next = self.x_prev + x_dot * dt
        # Update the previous state for the next step
        self.x_prev = x_next
        return x_next


    def create_dynamics(self):
        """
        Main dynamics function for integrating the complete AUV state.

        Args:
            t: Current time
            x: state space vector with [eta, nu, u]
            u_ref: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]

        Returns:
            state_vector_dot: Time derivative of complete state vector
        """
        x_sym = self.iX.sym('x', 13)
        u_sym = self.iX.sym('u', 6)
        self._dynamics_sym = cs.Function('dynamics', [x_sym, u_sym],
            [
                self.calculate_fx(x_sym) + self.calculate_gx(x_sym) @ u_sym 
            ]
        )
    def calculate_dynamics(self, x, u): 
        if not hasattr(self, '_dynamics_sym'):
            self.create_dynamics()
        if isinstance(x, np.ndarray):
            return np.array(self._dynamics_sym(x, u)).reshape((13,))
        else:
            return self._dynamics_sym(x, u)


    def create_fx(self):
        """
        Compute the autonomous dynamics of the AUV.
        Args:
            x: state space vector with [eta, nu]
        
        Returns:
            fx: Time derivative of the state space vector
        """
        eta_sym = self.iX.sym('eta', 7)
        nu_sym = self.iX.sym('nu', 6)
        nu_r_sym = self.iX.sym('nu_r', 6)
        euler = self.iX.sym('euler', 3)
        self._fx_sym = cs.Function('fx', [eta_sym, nu_sym, nu_r_sym, euler],
            [
                cs.vertcat(
                    self.calculate_eta_dynamics(eta_sym, nu_sym),
                    self.Minv @ (-cs.mtimes(self.calculate_C(nu_r=nu_r_sym), nu_r_sym) - \
                                    cs.mtimes(self.calculate_D(nu_r=nu_r_sym), nu_r_sym) - self.calculate_g(euler=euler))
                )
            ]
        )
    def calculate_fx(self, x):
        if not hasattr(self, '_fx_sym'):
            self.create_fx()
        eta = x[0:7]
        nu = x[7:13]
        nu_r, euler = self.calculate_system_state(eta, nu)
        if isinstance(x, np.ndarray):
            return np.array(self._fx_sym(eta, nu, nu_r, euler)).reshape((13,))
        else:
            return self._fx_sym(eta, nu, nu_r, euler)

    def create_gx(self):
        """
        Compute the control input matrix for the AUV.
        Args:
            x: state space vector with [eta, nu]
        Returns:
            gx: Control input matrix
        """
        x_sym = self.iX.sym('x', 13)
        self._gx_sym = cs.Function('gx', [x_sym],
            [
                cs.vertcat(
                    self.iX.zeros(7,6),  # No control inputs for eta
                    self.Minv  # The control input matrix is the inverse of the mass matrix
                )
            ]
        )
    def calculate_gx(self, x):
        if not hasattr(self, '_gx_sym'):
            self.create_gx()
        if isinstance(x, np.ndarray):
            return np.array(self._gx_sym(x)).reshape((x.shape[0], 6))
        else:
            return self._gx_sym(x)


    def create_system_state(self):
        """
        Extract speeds etc. based on state and control inputs
        """
        eta_sym = self.iX.sym('eta', 7)
        nu_sym = self.iX.sym('nu', 6)
        self._nu_r_euler_sym = cs.Function('nu_r_euler', [eta_sym, nu_sym],
            [
                nu_sym - cs.vertcat(
                    self.V_c * cs.cos(self.beta_c - quaternion_to_angles_cs(eta_sym[3:7])[0]),
                    self.V_c * cs.sin(self.beta_c - quaternion_to_angles_cs(eta_sym[3:7])[0]),
                    0,0,0,0
                ),
                quaternion_to_angles_cs(eta_sym[3:7]/cs.norm_2(eta_sym[3:7]))
            ]
        )
    def calculate_system_state(self, eta, nu):
        if not hasattr(self, '_nu_r_euler_sym'):
            self.create_system_state()
        if isinstance(eta, np.ndarray) and isinstance(nu, np.ndarray):
            nu_r, euler = self._nu_r_euler_sym(eta, nu)
            return np.array(nu_r).reshape((6,)), np.array(euler).reshape((3,))
        else:
            nu_r, euler = self._nu_r_euler_sym(eta, nu)
            return nu_r, euler


    def create_C(self):
        """
        Calculate Corriolis Matrix
        """
        # Define symbolic variables
        nu_r_sym = self.iX.sym('nu_r', 6)
        # Calculate the Coriolis matrix
        CRB = m2c_cs(self.MRB, nu_r_sym, self.iX)
        CA = m2c_cs(self.MA, nu_r_sym, self.iX)
        self._C_sym = cs.Function('C', [nu_r_sym], 
            [
                CRB + CA
            ]
        )
    def calculate_C(self, nu_r):
        if not hasattr(self, '_C_sym'):
            self.create_C()
        if isinstance(nu_r, np.ndarray):
            return np.array(self._C_sym(nu_r)).reshape((nu_r.shape[0], nu_r.shape[0]))
        else:
            return self._C_sym(nu_r)


    def create_D(self):
        """
        Calculate damping
        """
        nu_r_sym = self.iX.sym('nu_r', 6)
        self._D_sym = cs.Function('D', [nu_r_sym],
            [
                2*cs.diag(cs.vertcat(
                    self.Xuu * cs.fabs(nu_r_sym[0]),
                    self.Yvv * cs.fabs(nu_r_sym[1]),
                    self.Zww * cs.fabs(nu_r_sym[2]),
                    self.Kpp * cs.fabs(nu_r_sym[3]),
                    self.Mqq * cs.fabs(nu_r_sym[4]),
                    self.Nrr * cs.fabs(nu_r_sym[5])
                ))
            ]
        )
    def calculate_D(self, nu_r):
        if not hasattr(self, '_D_sym'):
            self.create_D()
        if isinstance(nu_r, np.ndarray):
            return np.array(self._D_sym(nu_r)).reshape((nu_r.shape[0], nu_r.shape[0]))
        else:
            return self._D_sym(nu_r)


    def create_g(self):
        """
        Calculate gravity vector
        """
        euler_sym = self.iX.sym('euler', 3)  # [psi, theta, phi]
        self._g_sym = cs.Function('g', [euler_sym],
            [
                gvect_cs(self.W, self.B, euler_sym[1], euler_sym[2], self.p_OG_O, self.p_OB_O)
            ]
        )
    def calculate_g(self, euler):
        if not hasattr(self, '_g_sym'):
            self.create_g()
        if isinstance(euler, np.ndarray):
            return np.array(self._g_sym(euler)).reshape((6,))
        else:
            return self._g_sym(euler)


    def create_eta_dynamics(self):
        """
        Computes the time derivative of position and quaternion orientation.

        Args:
            eta: [x, y, z, q0, q1, q2, q3] - Position and quaternion
            nu: [u, v, w, p, q, r] - Body-fixed velocities

        Returns:
            eta_dot: [ẋ, ẏ, ż, q̇0, q̇1, q̇2, q̇3]
        """
        eta = self.iX.sym('eta', 7)
        nu = self.iX.sym('nu', 6)
        # Extract position and quaternion
        q = eta[3:7]  # [q0, q1, q2, q3] where q0 is scalar part
        q = q/cs.norm_2(q)

        # Convert quaternion to DCM for position kinematics
        C = quaternion_to_dcm_cs(q)

        # Position dynamics: ṗ = C * v
        pos_dot = C @ nu[0:3]

        ## From Fossen 2021, eq. 2.78:
        om = nu[3:6]  # Angular velocity
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        # T_q_n_b = 0.5 * self.iX([
        #                         [-q1, -q2, -q3],
        #                         [q0, -q3, q2],
        #                         [q3, q0, -q1],
        #                         [-q2, q1, q0]
        #                         ])
        T_q_n_b = self.iX.zeros(4,3)
        T_q_n_b[0, 0] = -0.5*q1
        T_q_n_b[0, 1] = -0.5*q2
        T_q_n_b[0, 2] = -0.5*q3
        T_q_n_b[1, 0] = 0.5*q0
        T_q_n_b[1, 1] = -0.5*q3
        T_q_n_b[1, 2] = 0.5*q2
        T_q_n_b[2, 0] = 0.5*q3
        T_q_n_b[2, 1] = 0.5*q0
        T_q_n_b[2, 2] = -0.5*q1
        T_q_n_b[3, 0] = -0.5*q2
        T_q_n_b[3, 1] = 0.5*q1
        T_q_n_b[3, 2] = 0.5*q0

        q_dot = cs.mtimes(T_q_n_b, om) + self.gamma/2 * (1 - q.T @ q) * q

        self._eta_dynamics_sym = cs.Function('eta_dynamics', [eta, nu], 
            [
                cs.vertcat(pos_dot, q_dot)
            ]
        )
    def calculate_eta_dynamics(self, eta, nu):
        if not hasattr(self, '_eta_dynamics_sym'):
            self.create_eta_dynamics()
        if isinstance(eta, np.ndarray):
            return np.array(self._eta_dynamics_sym(eta, nu)).reshape((7,))
        else:
            return self._eta_dynamics_sym(eta, nu)


