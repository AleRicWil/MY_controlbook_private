import numpy as np
import blockbeamParam as P

class blockbeamCtrlPD:
    def __init__(self, alpha=0.0):
        # Desired responses
        tr_th = 0.2 # s, inner loop rise time
        zeta_th = 1/np.sqrt(2) # inner damping ratio
        wn_th = np.pi / (2*tr_th*np.sqrt(1-zeta_th**2))

        tr_z = 10*tr_th # s, outer loop rise time
        zeta_z = 1/np.sqrt(2) # outer damping ratio
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        # Initialize Self
        self.m1 = P.m1
        self.m2 = P.m2
        self.L = P.length
        self.g = P.g
        self.F_max = P.F_max
        # Define PD characteristic polynomical
        b0_th = P.length / ((self.m2*self.L**2/3) + self.m1*P.z0**2)
        a0_th = 0.0
        a1_th = 0.0

        b0_z = -self.g
        a0_z = 0.0
        a1_z = 0.0
        # PD gains
        self.kp_th = (wn_th**2 -a0_th) / b0_th #1.8251
        self.kd_th = (2*zeta_th*wn_th -a1_th) / b0_th #1.173
        self.kp_z = (wn_z**2 -a0_z) / b0_z #-0.0049
        self.kd_z = (2*zeta_z*wn_z -a1_z) / b0_z #-0.0317
        print('kp_th: ', self.kp_th)
        print('kd_th: ', self.kd_th)
        print('kp_z: ', self.kp_z)
        print('kd_z: ', self.kd_z)
        
    def update(self, z_ref, state):
        z = state[0][0]
        theta = state[1][0]
        zdot = state[2][0]
        thetadot = state[3][0]

        # Control input using PD control
        F_e = (self.m1*z + self.m2*self.L/2)*self.g / self.L
        theta_ref = self.kp_z * (z_ref-z) - self.kd_z*zdot  # Inner loop
        F_tilde = self.kp_th * (theta_ref - theta) - self.kd_th * thetadot # Outer loop
        
        tau = saturate(F_e + F_tilde, self.F_max)
        return tau
    
def saturate(u, limit):
    if abs(u) > limit:
        u = limit*np.sign(u)
    return u

