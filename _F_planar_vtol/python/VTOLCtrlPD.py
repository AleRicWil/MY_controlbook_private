import numpy as np
import VTOLParam as P

class VTOLCtrlPD:
    def __init__(self, alpha=0.0):
        # Desired responses
        tr_l = 0.5 # s, long rise time
        zeta_l = 1/np.sqrt(2) # long damping ration
        natFreq_l = np.pi / (2*tr_l*np.sqrt(1-zeta_l**2))

        tr_i = 0.1 # s, lat inner loop rise time
        zeta_i = 1/np.sqrt(2) # lat inner damping ratio
        natFreq_i = np.pi / (2*tr_i*np.sqrt(1-zeta_i**2))

        tr_o = 10*tr_i # s, lat outer loop rise time
        zeta_o = 1/np.sqrt(2) # lat outer damping ratio
        natFreq_o = np.pi / (2*tr_o*np.sqrt(1-zeta_o**2))
        # Define PD characteristic polynomical
        b0l = 1 / (P.mc + 2*P.mr)
        a0l = 0.0
        a1l = 0.0
        
        b0i = 1 / (P.Jc + 2*P.mr*P.d**2)
        a0i = 0.0
        a1i = 0.0

        F_e = (P.mc + 2*P.mr)*P.g
        b0o = -(F_e) / (P.mc * 2*P.mr)
        a0o = 0.0
        a1o = P.mu / (P.mc * 2*P.mr)
        # Solve for gains
        self.kp_h = (natFreq_l**2 -a0l) / b0l           #0.1134
        self.kd_h =  (2*zeta_l*natFreq_l -a1l) / b0l    #0.5833
        self.kp_th = (natFreq_i**2 -a0i) / b0i          #1.8251
        self.kd_th = (2*zeta_i*natFreq_i -a1i) / b0i    #1.173
        self.kp_z = (natFreq_o**2 -a0o) / b0o           #-0.0049
        self.kd_z = (2*zeta_o*natFreq_o -a1o) / b0o     #-0.0317
        # Assign parameters
        self.Jc = P.Jc
        self.mc = P.mc
        self.mr = P.mr
        self.d = P.d
        self.mu = P.mu
        self.g = P.g
        self.F_max = P.F_max
        # PD gains
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

        z_r = state_ref[0][0]
        h_r = state_ref[1][0]

        # compute the force using PD control
        F_equil = (self.mc + 2*self.mr)*self.g
        F_tilde = self.kp_h * (h_r - h) - self.kd_h * hdot
        F_tot = F_equil + F_tilde

        theta_r = self.kp_z * (z_r-z) - self.kd_z*zdot  # Inner loop
        torque =  self.kp_th * (theta_r-theta) - self.kd_th*thetadot
        
        F_r = F_tot/2 + (torque/2) / self.d
        F_L = F_tot/2 - (torque/2) / self.d
        
        tau = np.array([[F_r], [F_L]])
        return saturate(tau, self.F_max)
    
def saturate(u, limit):
    if u[0][0] < 0:
        u[0][0] = 0
    if u[1][0] < 0:
        u[1][0] = 0

    if u[0][0] > limit:
        u[0][0] = limit
    if u[1][0] > limit:
        u[1][0] = limit

    return u

