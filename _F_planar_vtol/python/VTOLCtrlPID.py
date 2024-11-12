import numpy as np
import VTOLParam as P

class VTOLCtrlPID:
    def __init__(self, alpha=0.0):
        # Desired responses
        tr_h = 1.2 # s, longitudinal rise time
        zeta_h = 1/np.sqrt(2) # longitudinal damping ration
        wn_h = np.pi / (2*tr_h*np.sqrt(1-zeta_h**2)) # longitudinal nat freq

        tr_th = 0.15 # s, lateral inner loop rise time
        zeta_th = 1/np.sqrt(2) # lateral inner damping ratio
        wn_th = np.pi / (2*tr_th*np.sqrt(1-zeta_th**2)) # lateral inner nat freq

        tr_z = 12*tr_th # s, lateral outer loop rise time
        zeta_z = 0.85#1/np.sqrt(2) # lateral outer damping ratio
        natFreq_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2)) # lateral outer nat frew
        # Initialize Self
        self.Jc = P.Jc
        self.mc = P.mc
        self.mr = P.mr
        self.d = P.d
        self.mu = P.mu
        self.g = P.g
        self.F_max = P.F_max
        self.theta_max = np.radians(30)
        # Define PID characteristic polynomical
        b0_h = 1 / (self.mc + 2*self.mr)
        a0_h = 0.0
        a1_h = 0.0
        
        b0_th = 1 / (self.Jc + 2*self.mr*self.d**2)
        a0_th = 0.0
        a1_th = 0.0

        F_e = (self.mc + 2*self.mr)*self.g
        b0_z = -(F_e) / (self.mc * 2*self.mr)
        a0_z = 0.0
        a1_z = self.mu / (self.mc * 2*self.mr)
        # PID gains
        self.kp_h = (wn_h**2 -a0_h) / b0_h           
        self.kd_h =  (2*zeta_h*wn_h -a1_h) / b0_h    
        self.kp_th = (wn_th**2 -a0_th) / b0_th         
        self.kd_th = (2*zeta_th*wn_th -a1_th) / b0_th    
        self.kp_z = (natFreq_z**2 -a0_z) / b0_z           
        self.kd_z = (2*zeta_z*natFreq_z -a1_z) / b0_z    
        self.ki_h = 1.5
        self.hdot_min = 0.1
        self.ki_z = -0.03
        self.zdot_min = 0.2
        print('kp_h: ', self.kp_h)
        print('ki_h: ', self.ki_h)
        print('kd_h: ', self.kd_h)
        print('kp_th: ', self.kp_th)
        print('kd_th: ', self.kd_th)
        print('kp_z: ', self.kp_z)
        print('ki_z: ', self.ki_z)
        print('kd_z: ', self.kd_z)

        # dirty derivative and integrator
        self.sigma = 0.01

        self.h_dot = P.hdot0            # estimated derivated of h
        self.z_dot = P.zdot0            # estimated derivative of z
        self.theta_dot = P.thetadot0    # estimated derivative of theta

        self.h_d1 = P.h0                # h delayed by one sample
        self.z_d1 = P.z0                # z delayed by one sample
        self.theta_d1 = P.theta0        # theta delayed by one sample

        self.error_h_dot = 0.0          # estimated derivated of h error
        self.error_z_dot = 0.0          # estimated derivative of error
        self.error_h_d1 = 0.0           # Error delayed by one sample
        self.error_z_d1 = 0.0           # Error delayed by one sample
        self.integrator_h = 0.0         # h integrator
        self.integrator_z = 0.0         # z integrator

    def update(self, state_ref, state):
        z = state[0][0]
        h = state[1][0]
        theta = state[2][0]

        z_ref = state_ref[0][0]
        h_ref = state_ref[1][0]

        # Longitudinal
        ''' PID control for (h_ref - h) to F '''
        # Calc h PID parameters
            # Proportional
        error_h = h_ref - h
            # dirty derivative
        self.h_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.h_dot \
                     + (2.0 / (2.0*self.sigma + P.Ts)) * ((h - self.h_d1))
            # anti-windup integral
        if abs(self.h_dot) < self.hdot_min:
            self.integrator_h = self.integrator_h + (P.Ts / 2) * (error_h + self.error_h_d1)
            print('h: ', self.integrator_h)
        else:
            print('.')
            pass

        # F from h PID control
        F_e = (self.mc + 2*self.mr)*self.g
        F_tilde = self.kp_h*error_h + self.ki_h*self.integrator_h - self.kd_h*self.h_dot
        F = F_e + F_tilde
            
        # Lateral
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
            print('z: ', self.integrator_z)
        else:
            print('.')

        # theta_ref from z PID control
        theta_ref = self.kp_z*error_z + self.ki_z*self.integrator_z - self.kd_z*self.z_dot
        theta_ref = saturate_theta(theta_ref, self.theta_max)

        ''' PD control for (theta_ref - theta) to F '''
        # Calc theta PD parameters
            # Proportional
        error_th = theta_ref - theta
            # dirty dertivative
        self.theta_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.theta_dot \
            + (2.0 / (2.0*self.sigma + P.Ts)) * ((theta - self.theta_d1))
        
        # torque from theta PD control
        torque =  self.kp_th*error_th - self.kd_th*self.theta_dot

        '''Combine to individual motor forces'''
        F_r = F/2 + (torque/2) / self.d
        F_L = F/2 - (torque/2) / self.d
        
        tau = saturate(np.array([[F_r], [F_L]]), self.F_max)

        '''update delayed variables'''
        self.error_h_d1 = error_h
        self.error_z_d1 = error_z
        self.h_d1 = h
        self.z_d1 = z
        self.theta_d1 = theta
        return tau
    
def saturate(u, limit):
    if u[0][0] < 0:
        u[0][0] = 0
    elif u[0][0] > limit:
        u[0][0] = limit

    if u[1][0] < 0:
        u[1][0] = 0
    elif u[1][0] > limit:
        u[1][0] = limit

    return u

def saturate_theta(theta, limit):
    if abs(theta) > limit:
        theta = limit*np.sign(theta)
    return theta