import numpy as np
import blockbeamParam as P

class blockbeamCtrlPD:
    def __init__(self, alpha=0.0):
        # Desired responses
        tr_i = 0.2 # s, inner loop rise time
        zeta_i = 1/np.sqrt(2) # inner damping ratio
        natFreq_i = np.pi / (2*tr_i*np.sqrt(1-zeta_i**2))

        tr_o = 10*tr_i # s, outer loop rise time
        zeta_o = 1/np.sqrt(2) # outer damping ratio
        natFreq_o = np.pi / (2*tr_o*np.sqrt(1-zeta_o**2))
        # Define PD characteristic polynomical
        b0i = P.length / ((P.m2*P.length**2/3) + P.m1*P.z0**2)
        a0i = 0.0
        a1i = 0.0

        b0o = -P.g
        a0o = 0.0
        a1o = 0.0
        # Solve for gains
        self.kp_th = (natFreq_i**2 -a0i) / b0i #1.8251
        self.kd_th = (2*zeta_i*natFreq_i -a1i) / b0i #1.173
        self.kp_z = (natFreq_o**2 -a0o) / b0o #-0.0049
        self.kd_z = (2*zeta_o*natFreq_o -a1o) / b0o #-0.0317
        self.m1 = P.m1
        self.m2 = P.m2
        self.L = P.length
        self.g = P.g
        self.F_max = P.F_max
        # PD gains
        print('kp_th: ', self.kp_th)
        print('kd_th: ', self.kd_th)
        print('kp_z: ', self.kp_z)
        print('kd_z: ', self.kd_z)


    def update(self, z_r, state):
        z = state[0][0]
        theta = state[1][0]
        zdot = state[2][0]
        thetadot = state[3][0]

        
        
        # compute the force using PD control
        F_equil = (self.m1*z + self.m2*self.L/2)*self.g / self.L
        theta_r = self.kp_z * (z_r-z) - self.kd_z*zdot  # Inner loop
        F_tilde = self.kp_th * (theta_r - theta) - self.kd_th * thetadot # Outer loop
        
        tau = saturate(F_equil + F_tilde, self.F_max)
        return tau
    
def saturate(u, limit):
    if abs(u) > limit:
        u = limit*np.sign(u)
        # print('sat')
    return u

