import numpy as np
import blockbeamParam as P

class blockbeamCtrlPID:
    def __init__(self, alpha=0.0):
        # Desired responses
        tr_th = 0.2 # s, inner loop rise time
        zeta_th = 1/np.sqrt(2) # inner damping ratio
        natFreq_th = np.pi / (2*tr_th*np.sqrt(1-zeta_th**2))

        tr_z = 10*tr_th # s, outer loop rise time
        zeta_z = 1/np.sqrt(2) # outer damping ratio
        natFreq_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        # Initialize Self
        self.m1 = P.m1
        self.m2 = P.m2
        self.L = P.length
        self.g = P.g
        self.F_max = P.F_max
        self.Theta_max = P.Theta_max
        # Define PID characteristic polynomials
        b0_th = self.L / ((self.m2*self.L**2/3) + self.m1*P.z0**2)
        a0_th = 0.0
        a1_th = 0.0

        b0_z = -self.g
        a0_z = 0.0
        a1_z = 0.0
        # PID gains
        self.kp_th = (natFreq_th**2 -a0_th) / b0_th          
        self.kd_th = (2*zeta_th*natFreq_th -a1_th) / b0_th    
        self.kp_z = (natFreq_z**2 -a0_z) / b0_z          
        self.kd_z = (2*zeta_z*natFreq_z -a1_z) / b0_z     
        self.ki_z = -0.06
        self.zdot_min = 0.03    # anti-windup threshold
        print('kp_th: ', self.kp_th)
        print('kd_th: ', self.kd_th)
        print('kp_z: ', self.kp_z)
        print('ki_z: ', self.ki_z)
        print('kd_z: ', self.kd_z)
        
        # dirty derivative and integrator
        self.sigma = 0.02

        self.z_dot = P.zdot0            # estimated derivative of z
        self.theta_dot = P.thetadot0    # estimated derivative of theta
        self.z_d1 = P.z0                # z delayed by one sample
        self.theta_d1 = P.theta0        # theta delayed by one sample
        self.error_z_dot = 0.0          # estimated derivative of error
        self.error_z_d1 = 0.0           # Error delayed by one sample
        self.integrator_z = 0.0         # integrator


    def update(self, z_ref, state):
        z = state[0][0]
        theta = state[1][0]

        ''' PID control for (z_ref - z) to theta_r '''
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

        # theta_ref from z PID control
        theta_ref = self.kp_z*error_z + self.ki_z*self.integrator_z - self.kd_z*self.z_dot
            
        ''' PD control for (theta_ref - theta) to F '''
        # Calc theta PD parameters
            # Proportional
        error_th = theta_ref - theta
            # dirty dertivative
        self.theta_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.theta_dot \
            + (2.0 / (2.0*self.sigma + P.Ts)) * ((theta - self.theta_d1))
        
        # F from theta PD control
        F_e = (self.m1*z + self.m2*self.L/2)*self.g / self.L
        F_tilde = self.kp_th*error_th - self.kd_th*self.theta_dot
        
        tau = saturate(F_e + F_tilde, self.F_max)
        
        '''update delayed variables'''
        self.error_z_d1 = error_z
        self.z_d1 = z
        self.theta_d1 = theta

        return tau
    
def saturate(u, limit):
    if abs(u) > limit:
        u = limit*np.sign(u)
        # print('sat')
    return u

