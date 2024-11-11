import numpy as np
import VTOLParam as P

class VTOLCtrlPD:
    def __init__(self, alpha=0.0):
        # Desired responses
        tr_h = 1.2 # s, longitudinal rise time
        zeta_h = 1/np.sqrt(2) # longitudinal damping ration
        wn_h = np.pi / (2*tr_h*np.sqrt(1-zeta_h**2)) # longitudinal nat freq

        tr_th = 0.1 # s, lateral inner loop rise time
        zeta_th = 1/np.sqrt(2) # lateral inner damping ratio
        wn_th = np.pi / (2*tr_th*np.sqrt(1-zeta_th**2)) # lateral inner nat freq

        tr_z = 15*tr_th # s, lateral outer loop rise time
        zeta_z = 1/np.sqrt(2) # lateral outer damping ratio
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
        # Define PD characteristic polynomical
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
        # PD gains
        self.kp_h = (wn_h**2 -a0_h) / b0_h           
        self.kd_h =  (2*zeta_h*wn_h -a1_h) / b0_h    
        self.kp_th = (wn_th**2 -a0_th) / b0_th         
        self.kd_th = (2*zeta_th*wn_th -a1_th) / b0_th    
        self.kp_z = (natFreq_z**2 -a0_z) / b0_z           
        self.kd_z = (2*zeta_z*natFreq_z -a1_z) / b0_z     
        print('kp_h: ', self.kp_h)
        print('kd_h: ', self.kd_h)
        print('kp_th: ', self.kp_th)
        print('kd_th: ', self.kd_th)
        print('kp_z: ', self.kp_z)
        print('kd_z: ', self.kd_z)

    def update(self, state_ref, state):
        z = state[0][0]
        h = state[1][0]
        theta = state[2][0]
        zdot = state[3][0]
        hdot = state[4][0]
        thetadot = state[5][0]

        z_ref = state_ref[0][0]
        h_ref = state_ref[1][0]

        # Control input using PD control
            # Longitudinal
        F_e = (self.mc + 2*self.mr)*self.g
        F_tilde = self.kp_h * (h_ref - h) - self.kd_h * hdot
        F = F_e + F_tilde
            # Lateral
        theta_ref = self.kp_z * (z_ref-z) - self.kd_z*zdot  # Inner loop
        theta_ref = saturate_theta(theta_ref, self.theta_max)
        torque =  self.kp_th * (theta_ref-theta) - self.kd_th*thetadot # Outer loop
            # Combine to individual motor forces
        F_r = F/2 + (torque/2) / self.d
        F_L = F/2 - (torque/2) / self.d
        
        tau = saturate(np.array([[F_r], [F_L]]), self.F_max)
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