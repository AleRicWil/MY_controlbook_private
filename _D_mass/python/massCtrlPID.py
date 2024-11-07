import numpy as np
import massParam as P

class massCtrlPID:
    def __init__(self, alpha=0.0):
        # Desired responses
        tr = 2 # s, inner loop rise time
        zeta = 1/np.sqrt(2) # inner damping ratio
        natFreq = np.pi / (2*tr*np.sqrt(1-zeta**2))
        # Define PD characteristic polynomical
        b0 = 1 / P.m
        a0 = P.k / P.m
        a1 = P.b / P.m
        # PID gains
        self.kp = (natFreq**2 -a0) / b0 #1.8251
        self.kd = (2*zeta*natFreq -a1) / b0 #1.173
        self.ki = 1.0
        print('kp: ', self.kp)
        print('ki: ', self.ki)
        print('kd: ', self.kd)
        # Initialize self.
        self.m = P.m
        self.k = P.k
        self.b = P.b
        self.F_max = P.F_max
        # dirty derivative and integrator
        self.sigma = 0.05
        self.z_dot = P.zdot0  # estimated derivative of z
        self.z_d1 = P.z0  # theta delayed by one sample
        self.error_dot = 0.0  # estimated derivative of error
        self.error_d1 = 0.0  # Error delayed by one sample
        self.integrator = 0.0  # integrator

    def update(self, z_r, state):
        z = state[0][0]
        zdot = state[1][0]
    
        # Proportional to error
        error = z_r - z
        # dirty dertivative of position
        self.z_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.z_dot \
                     + (2.0 / (2.0*self.s4igma + P.Ts)) * ((z - self.z_d1))
        # anti-windup integral of error
        if abs(self.z_dot) < 0.1:
            self.integrator = self.integrator + (P.Ts / 2) * (error + self.error_d1)
        
        # PID control input
        F_tilde = self.kp*error + self.ki*self.integrator - self.kd*self.z_dot

        # compute the force using PD control
        F_e = self.k * z_r
        tau = saturate(F_e + F_tilde, self.F_max)
        # update delayed variables
        self.error_d1 = error
        self.z_d1 = z
        return tau
    
def saturate(u, limit):
    if abs(u) > limit:
        u = limit*np.sign(u)
    return u

