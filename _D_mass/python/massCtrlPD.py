import numpy as np
import massParam as P

class massCtrlPD:
    def __init__(self, alpha=0.0):
        # Desired responses
        tr_z = 2.0 # s, inner loop rise time
        zeta_z = 1/np.sqrt(2) # inner damping ratio
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        # Initialize self.
        self.m = P.m
        self.k = P.k
        self.b = P.b
        self.F_max = P.F_max
        # Define PID characteristic polynomial
        b0_z = 1 / self.m
        a0_z = self.k / self.m
        a1_z = self.b / self.m
        # PID gains
        self.kp_z = (wn_z**2 -a0_z) / b0_z 
        self.kd_z = (2*zeta_z*wn_z -a1_z) / b0_z
        print('kp: ', self.kp_z)
        print('kd: ', self.kd_z)


    def update(self, z_r, state):
        z = state[0][0]
        zdot = state[1][0]
        
        # compute the force using PD control
        F_e = self.k * z_r
        F_tilde = self.kp_z * (z_r - z) - self.kd_z * zdot
        tau = saturate(F_e + F_tilde, self.F_max)

        return tau
    
def saturate(u, limit):
    if abs(u) > limit:
        u = limit*np.sign(u)
        print('sat')
    return u

