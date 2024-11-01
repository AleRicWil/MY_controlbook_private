import numpy as np
import massParam as P

class massCtrlPD:
    def __init__(self, alpha=0.0):
        # Declare pole locations
        # p1 = -3
        # p2 = -4
        # alpha0 = (-p1) * (-p2) # Coefficient in front of s^0 in desired characteristic equation
        # alpha1 = (-p1) + (-p2) # Coefficient in front of s^1 in desired characteristic equation
        # b0 = 3 / (P.m * P.ell**2)
        # a0 = 0.0
        # a1 = 3 * P.b / (P.m * P.ell**2)
        self.kp = 3.05#(alpha0 - a0) / b0
        self.kd = 7.2#(alpha1 - a1) / b0
        self.k = P.k
        self.F_max = P.F_max
        # PD gains
        print('kp: ', self.kp)
        print('kd: ', self.kd)


    def update(self, z_r, state):
        z = state[0][0]
        zdot = state[1][0]
        kp = 3.1#3.05
        kd = 7.7#7.20
        
        # compute the force using PD control
        tau_equil = self.k * z_r
        tau_tilde = self.kp * (z_r - z) - self.kd * zdot
        tau = saturate(tau_equil + tau_tilde, self.F_max)
        return tau
    
def saturate(u, limit):
    if abs(u) > limit:
        u = limit*np.sign(u)
        print('sat')
    return u

