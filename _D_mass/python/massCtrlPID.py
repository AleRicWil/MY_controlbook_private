import numpy as np
import massParam as P

class massCtrlPID:
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
        b0_z = 1 / P.m
        a0_z = P.k / P.m
        a1_z = P.b / P.m
        # PID gains
        self.kp_z = (wn_z**2 -a0_z) / b0_z 
        self.kd_z = (2*zeta_z*wn_z -a1_z) / b0_z 
        self.ki_z = 2.2
        self.zdot_min = 0.1     # anti-windup threshold
        print('kp: ', self.kp_z)
        print('ki: ', self.ki_z)
        print('kd: ', self.kd_z)
        
        # dirty derivative and integrator
        self.sigma = 0.02
        self.z_dot = P.zdot0    # estimated derivative of z
        self.z_d1 = P.z0        # z delayed by one sample
        self.error_z_d1 = 0.0   # z error delayed by one sample
        self.integrator_z = 0.0 # z integrator

    def update(self, z_ref, state):
        z = state[0][0]
    
        '''PID control for (z_ref - z) to F'''
        # Calc z PID parameters
            # Proportional
        error_z = z_ref - z
            # dirty dertivative
        self.z_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.z_dot \
                     + (2.0 / (2.0*self.sigma + P.Ts)) * ((z - self.z_d1))
            # anti-windup integral
        if abs(self.z_dot) < self.zdot_min:
            self.integrator_z = self.integrator_z + (P.Ts / 2) * (error_z + self.error_z_d1)
            print(self.integrator_z)
        else:
            print('.')
        # F from z PID control
        F_e = self.k * z_ref
        F_tilde = self.kp_z*error_z + self.ki_z*self.integrator_z - self.kd_z*self.z_dot
        
        tau = saturate(F_e + F_tilde, self.F_max)
        
        '''update delayed variables'''
        self.error_z_d1 = error_z
        self.z_d1 = z
        
        return tau
    
def saturate(u, limit):
    if abs(u) > limit:
        u = limit*np.sign(u)
    return u

