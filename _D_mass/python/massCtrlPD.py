import numpy as np
import massParam as P

class massCtrlPD:
    def __init__(self):
        # Declare pole locations
        # p1 = -3
        # p2 = -4
        # alpha0 = (-p1) * (-p2) # Coefficient in front of s^0 in desired characteristic equation
        # alpha1 = (-p1) + (-p2) # Coefficient in front of s^1 in desired characteristic equation
        # b0 = 3 / (P.m * P.ell**2)
        # a0 = 0.0
        # a1 = 3 * P.b / (P.m * P.ell**2)
        self.kp = 4.5 #(alpha0 - a0) / b0
        self.kd = 12 #(alpha1 - a1) / b0
        self.k = P.k
        # PD gains
        print('kp: ', self.kp)
        print('kd: ', self.kd)


    def update(z_r, x):
        z = x[0][0]
        zdot = x[1][0]
        kp = 3.1#3.05
        kd = 7.7#7.20
        
        # compute the force using PD control
        f_equil = P.k * z_r
        f_gain = kp * (z_r - z) - kd * zdot

        return saturate(f_equil + f_gain)
    
def saturate(f):
    if abs(f) > 6:
        f = 6*np.sign(f)
        print('sat')
    return f

