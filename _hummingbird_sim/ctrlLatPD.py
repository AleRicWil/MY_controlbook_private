import numpy as np
import hummingbirdParam as P
from ctrlLonPD import ctrlLonPD


class ctrlLatPD:
    def __init__(self):
        self.LonCtrl = ctrlLonPD()
        # Desired responses
        tr_phi = 0.25 # s, roll rise time
        zeta_phi = 1/np.sqrt(2) # roll damping ratio
        wn_phi = np.pi / (2*tr_phi*np.sqrt(1-zeta_phi**2)) # rad/s, roll nat freq
        
        tr_psi = tr_phi*10 # s, yaw rise time
        zeta_psi = 1/np.sqrt(2) # yaw damping ratio
        wn_psi = np.pi / (2*tr_psi*np.sqrt(1-zeta_psi**2)) # rad/s, yaw nat freq
        # Initialize self
        self.ell1 = P.ell1
        self.ell2 = P.ell2
        self.ell3x = P.ell3x
        self.ell3y = P.ell3y
        self.ell3z = P.ell3z
        self.ellT = P.ellT
        self.d = P.d
        self.m1 = P.m1
        self.m2 = P.m2
        self.m3 = P.m3
        self.J1x = P.J1x
        self.J1y = P.J1y
        self.J1z = P.J1z
        self.J2x = P.J2x
        self.J2y = P.J2y
        self.J2z = P.J2z 
        self.J3x = P.J3x 
        self.J3y = P.J3y 
        self.J3z = P.J3z
        self.JT = P.JT
        self.km = P.km
        self.g = P.g
        # Define PD characteristic polynomical
        b0_phi = 1/P.J1x
        a0_phi = 0.0
        a1_phi = 0.0

        self.F_e = (P.m1*P.ell1 + P.m2*P.ell2)*P.g / P.ellT
        b0_psi = (self.ellT*self.F_e)/(self.JT + self.J1z)
        a0_psi = 0.0
        a1_psi = 0.0
        # PD Gains
        self.kp_phi = (wn_phi**2 -a0_phi) / b0_phi           
        self.kd_phi =  (2*zeta_phi*wn_phi -a1_phi) / b0_phi    
        self.kp_psi = (wn_psi**2 -a0_psi) / b0_psi         
        self.kd_psi = (2*zeta_psi*wn_psi -a1_psi) / b0_psi
        print('kp_roll: ', self.kp_phi)
        print('kd_roll: ', self.kd_phi) 
        print('kp_yaw: ', self.kp_psi)
        print('kd_yaw: ', self.kd_psi) 
        # sample rate of the controller
        self.Ts = P.Ts
        # dirty derivative parameters
        self.sigma = 0.05  # cutoff freq for dirty derivative
        # delayed variables
        self.phi_d1 = P.phi0
        self.psi_d1 = P.psi0
        self.phi_dot = P.phidot0
        self.psi_dot = P.psidot0
        self.error_phi_d1 = 0.0  # pitch error delayed by 1
        self.effor_psi_d1 = 0.0


    def update(self, psi_ref, state: np.ndarray):
        phi = state[0][0]
        psi = state[2][0]
        
        ''' PD Yaw Control for (psi_ref - psi) to phi_ref'''
        # Calc psi PD parameters
            # Proportional
        error_psi = psi_ref - psi
            # dirty derivative
        self.psi_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.psi_dot \
                         + (2.0 / (2.0*self.sigma + P.Ts)) * ((psi - self.psi_d1))
        phi_ref = self.kp_psi*error_psi - self.kd_psi*self.psi_dot
        
        '''PD Pitch Control for (phi_ref - phi) to torque'''
        # Calc phi PD parameters
            # Proportional
        error_phi = phi_ref - phi
            # dirty derivative
        self.phi_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.phi_dot \
                         + (2.0 / (2.0*self.sigma + P.Ts)) * ((phi - self.phi_d1))

        torque_e = 0.0
        torque_tilde = self.kp_phi*error_phi - self.kd_phi*self.phi_dot
        torque = torque_e + torque_tilde


        # convert force and torque to pwm signals
        # tau = np.array([[self.F_e + (torque / P.d)],   # u_left
        #                 [self.F_e - (torque / P.d)]])  # u_right      
        tau = np.array([[(torque / P.d)],   # u_left
                        [(-torque / P.d)]])  # u_right         
        pwm_Lat = tau / (2 * P.km)
        pwm_Lon = self.LonCtrl.update(0.0, state)
        pwm = saturate(pwm_Lat + pwm_Lon, 0, 1)
        # update all delayed variables
        self.phi_d1 = phi
        self.psi_d1 = psi
        self.error_phi_d1 = error_phi
        self.error_psi_d1 = error_psi
        # return pwm plus reference signals
        return pwm, phi_ref#, np.array([[0.], [theta_r], [0.]])


def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        if u > up_limit:
            u = up_limit
        if u < low_limit:
            u = low_limit
    else:
        for i in range(0, u.shape[0]):
            if u[i][0] > up_limit:
                u[i][0] = up_limit
            if u[i][0] < low_limit:
                u[i][0] = low_limit
    return u




