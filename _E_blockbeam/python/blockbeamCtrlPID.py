import numpy as np
import blockbeamParam as P

class blockbeamCtrlPID:
    def __init__(self, alpha=0.0):
        # Desired responses
        tr_i = 0.5 # s, inner loop rise time
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
        # PID gains
        self.kp_th = (natFreq_i**2 -a0i) / b0i          #1.8251
        self.kd_th = (2*zeta_i*natFreq_i -a1i) / b0i    #1.173
        self.kp_z = (natFreq_o**2 -a0o) / b0o           #-0.0049
        self.kd_z = (2*zeta_o*natFreq_o -a1o) / b0o     #-0.0317
        self.ki_z = -0.03
        print('kp_th: ', self.kp_th)
        print('kd_th: ', self.kd_th)
        print('kp_z: ', self.kp_z)
        print('ki_z: ', self.ki_z)
        print('kd_z: ', self.kd_z)
        # Initialize Self
        self.m1 = P.m1
        self.m2 = P.m2
        self.L = P.length
        self.g = P.g
        self.F_max = P.F_max
        self.Theta_max = P.Theta_max
        # dirty derivative and integrator
        self.sigma = 0.05

        self.z_dot = P.zdot0            # estimated derivative of z
        self.theta_dot = P.thetadot0    # estimated derivative of theta
        self.z_d1 = P.z0                # z delayed by one sample
        self.theta_d1 = P.theta0        # theta delayed by one sample
        self.error_z_dot = 0.0          # estimated derivative of error
        self.error_z_d1 = 0.0           # Error delayed by one sample
        self.integrator_z = 0.0         # integrator


    def update(self, z_r, state):
        z = state[0][0]
        theta = state[1][0]
        zdot = state[2][0]
        thetadot = state[3][0]

        # Proportional to error
        error_z = z_r - z
        # dirty dertivative of position
        self.z_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.z_dot \
                     + (2.0 / (2.0*self.sigma + P.Ts)) * ((z - self.z_d1))
        # anti-windup integral of error
        if abs(self.z_dot) < 0.01:
            print('here')
            print(self.z_dot)
            print(self.integrator_z)
            self.integrator_z = self.integrator_z + (P.Ts / 2) * (error_z + self.error_z_d1)
        # elif abs(self.z_dot) > 0.02:
        #     self.integrator_z = 0
        # PID control input
        theta_r = self.kp_z*error_z + self.ki_z*self.integrator_z - self.kd_z*self.z_dot
        # theta_r = saturate(theta_r, self.Theta_max)
        
        # Proportional to error
        error_th = theta_r - theta
        # dirty dertivative of theta
        self.theta_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.theta_dot \
            + (2.0 / (2.0*self.sigma + P.Ts)) * ((theta - self.theta_d1))
         # PD control on theta
        F_tilde = self.kp_th*error_th - self.kd_th*self.theta_dot

        # compute the force using PD control
        F_e = (self.m1*z + self.m2*self.L/2)*self.g / self.L
        tau = saturate(F_e + F_tilde, self.F_max)
        # update delayed variables
        self.error_z_d1 = error_z
        self.z_d1 = z
        self.theta_d1 = theta
        return tau
    
def saturate(u, limit):
    if abs(u) > limit:
        u = limit*np.sign(u)
        # print('sat')
    return u

