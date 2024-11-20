import numpy as np
import control as cnt
import VTOLParam as P
from scipy import signal

class ctrlObserver:
    def __init__(self):
        '''Full State Feedback'''
        # Desired responses
            # longitudinal
        tr_h = 1.2 # s, longitudinal rise time
        zeta_h = 1/np.sqrt(2) # longitudinal damping ration
        wn_h = np.pi / (2*tr_h*np.sqrt(1-zeta_h**2)) # longitudinal nat freq
        integrator_pole_h = -0.8
        self.hdot_min = 0.4
        
            # lateral
        tr_th = 0.15 # s, lateral inner loop rise time
        zeta_th = 1/np.sqrt(2) # lateral inner damping ratio
        wn_th = np.pi / (2*tr_th*np.sqrt(1-zeta_th**2)) # lateral inner nat freq

        tr_z = 15*tr_th # s, lateral outer loop rise time
        zeta_z = 1/np.sqrt(2) # lateral outer damping ratio
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2)) # lateral outer nat freg
        integrator_pole_z = -0.4
        self.zdot_min = 0.9
        # Initialize Self
        self.Jc = P.Jc
        self.mc = P.mc
        self.mr = P.mr
        self.d = P.d
        self.mu = P.mu
        self.g = P.g
        self.F_max = P.F_max
        self.theta_max = np.radians(30)

        # poles of characteristic polynomials
            # Longitudinal
        desCharPoly_lgtd = np.convolve([1, 2 * zeta_h * wn_h, wn_h**2],
                                       [1, -integrator_pole_h])
        desPoles_lgtd = np.roots(desCharPoly_lgtd)
            # Lateral
        TEMP_desCharPoly_lat = np.convolve([1, 2 * zeta_z * wn_z, wn_z**2],
                                           [1, 2 * zeta_th * wn_th, wn_th**2])
        desCharPoly_lat = np.convolve(TEMP_desCharPoly_lat,
                                      [1, -integrator_pole_z])
        desPoles_lat = np.roots(desCharPoly_lat)
        # State Space Equations
            # xdot = A*x + B*u
            #   y  = C*x
        p = self.mc + 2*self.mr
        Fe = p*self.g
        '''Longitudinal'''
        self.A_lgtd = np.array([[0.0, 1.0],
                                [0.0, 0.0]])

        self.B_lgtd = np.array([[0.0],
                                [1/p]])

        self.C_lgtd = np.array([[1.0, 0.0]])
        self.Cr_lgtd = self.C_lgtd
        # Integral Augmented system matrices
        A1_lgtd = np.vstack((np.hstack((self.A_lgtd  , np.zeros((len(self.A_lgtd) , 1)))),
                             np.hstack((-self.Cr_lgtd, np.zeros((len(self.Cr_lgtd), 1)))) ))
        B1_lgtd = np.vstack((self.B_lgtd, 0.0))

        '''Lateral'''
        self.A_lat = np.array([[0.0, 0.0    , 1.0       , 0.0],
                               [0.0, 0.0    , 0.0       , 1.0],
                               [0.0, -Fe/p  , -self.mu/p, 0.0],
                               [0.0, 0.0    , 0.0       , 0.0]])
        self.B_lat = np.array([[0.0],
                               [0.0],
                               [0.0],
                               [1/(self.Jc + 2*self.mr*self.d**2)]])        
        self.C_lat = np.array([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0]])
        self.Cr_lat = np.array([[1.0, 0.0, 0.0, 0.0]])
        # Integral Augmented system matrices
        A1_lat = np.vstack((np.hstack((self.A_lat  , np.zeros((len(self.A_lat) , 1)))),
                            np.hstack((-self.Cr_lat, np.zeros((len(self.Cr_lat), 1)))) ))
        B1_lat = np.vstack((self.B_lat, 0.0))

        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A1_lgtd, B1_lgtd)) != np.size(A1_lgtd,1)  or \
                np.linalg.matrix_rank(cnt.ctrb(A1_lat, B1_lat)) != np.size(A1_lat,1):
            print("The system is not controllable")
        else:
                # Longitudinal
            self.K1_lgtd = (cnt.place(A1_lgtd, B1_lgtd, desPoles_lgtd))
            self.K_lgtd = self.K1_lgtd[0,:-1]
            self.ki_lgtd = self.K1_lgtd[0][-1]
                # Lateral
            self.K1_lat = (cnt.place(A1_lat, B1_lat, desPoles_lat))
            self.K_lat = self.K1_lat[0,:-1]
            self.ki_lat = self.K1_lat[0][-1]


        '''Observer'''
            # Longitudinal
        tr_h_obs = tr_h/10
        zeta_h_obs = 1/np.sqrt(2)
        wn_h_obs = np.pi / (2*tr_h_obs*np.sqrt(1-zeta_h_obs**2))
        desCharPoly_lgtd_obs = np.array([1, 2 * zeta_h_obs * wn_h_obs, wn_h_obs**2])
        desPoles_lgtd_obs = np.roots(desCharPoly_lgtd_obs)

        if np.linalg.matrix_rank(cnt.ctrb(self.A_lgtd.T, self.C_lgtd.T)) != np.size(self.C_lgtd, axis=1):
            print("The system is not observerable, lateral")
        else:
            self.L_lgtd = cnt.acker(self.A_lgtd.T, self.C_lgtd.T, desPoles_lgtd_obs).T

            # Lateral
        tr_z_obs = tr_z/10
        zeta_z_obs = 1/np.sqrt(2)
        wn_z_obs = np.pi / (2*tr_z_obs*np.sqrt(1-zeta_z_obs**2))
        
        tr_th_obs = tr_th/10
        zeta_th_obs = 1/np.sqrt(2)
        wn_th_obs = np.pi / (2*tr_th_obs*np.sqrt(1-zeta_th_obs**2))
        
        desCharPoly_lat_obs = np.convolve([1, 2 * zeta_z_obs * wn_z_obs, wn_z_obs**2],
                                          [1, 2 * zeta_th_obs * wn_th_obs, wn_th_obs**2])
        desPoles_lat_obs = np.roots(desCharPoly_lat_obs)
        # Compute the observer gains if the system is observable
        if np.linalg.matrix_rank(cnt.ctrb(self.A_lat.T, self.C_lat.T)) != np.size(self.C_lat, axis=1):
            print("The system is not observable, longitudinal")
        else:
            self.L_lat = signal.place_poles(self.A_lat.T, self.C_lat.T, desPoles_lat_obs).gain_matrix.T


        print('K_lgtd: ', self.K_lgtd)
        print('ki_lgtd: ', self.ki_lgtd)
        print(f'desPoles_lgtd: {desPoles_lgtd}')
        print('L_lgtd.T: ', self.L_lgtd.T)
        print(f'desPoles_lgtd_obs: {desPoles_lgtd_obs}')
        
        print('\nK_lat: ', self.K_lat)
        print('ki_lat: ', self.ki_lat)
        print(f'desPoles_lat: {desPoles_lat}')
        print('L_lat.T ', self.L_lat)
        print(f'desPoles_lat_obs: {desPoles_lat_obs}\n\n')
         
         # integrator
        self.error_h_d1 = 0.0           # Error delayed by one sample
        self.error_z_d1 = 0.0           # Error delayed by one sample
        self.integrator_h = 0.0         # h integrator
        self.integrator_z = 0.0         # z integrator

        self.state_hat = np.array([[P.z0], [P.h0], [P.theta0], [0.0], [0.0], [0.0]])
        F_e = (self.mc + 2*self.mr)*self.g
        F_r = F_L = F_e/2
        self.tau_d1 = np.array([[F_r], [F_L]])

    def update(self, ref, y_measured):
        state_hat = self.update_observer(y_measured)
        state_hat_lgtd = np.array([[state_hat[1][0]], [state_hat[4][0]]])
        state_hat_lat = np.array([[state_hat[0][0]], [state_hat[2][0]], [state_hat[3][0]], [state_hat[5][0]]])
        h_hat = state_hat_lgtd[0][0]
        hdot_hat = state_hat_lgtd[1][0]
        z_hat = state_hat_lat[0][0]
        zdot_hat = state_hat_lat[2][0]

        z_ref = ref[0][0]
        h_ref = ref[1][0]

            # Longitudinal
        error_h = h_ref - h_hat
        if abs(hdot_hat) < self.hdot_min:
            self.integrator_h = self.integrator_h + (P.Ts / 2) * (error_h + self.error_h_d1)
            #print(f'h: ', self.integrator_h)
        else:
            #print('.')
            pass
        # Eqilibrium point at any value h. Use current h_ref
        state_tilde_lgtd = state_hat_lgtd - np.array([[h_ref], [0]])
        '''Feedback control for (h_ref - h_hat) to F'''
        F_e = (self.mc + 2*self.mr)*self.g
        F_tilde = -self.K_lgtd@state_tilde_lgtd - self.ki_lgtd*self.integrator_h
        F = F_e + F_tilde

            # Lateral
        error_z = z_ref - z_hat
        if abs(zdot_hat) < self.zdot_min:
            self.integrator_z = self.integrator_z + (P.Ts / 2) * (error_z + self.error_z_d1)
            #print(f'z: ', self.integrator_z)
        else:
            #print('.')
            pass
            
        '''Feedback control for (z_ref - z_hat) to torque'''
        state_tilde_lat = state_hat_lat - np.array([[z_ref], [0], [0], [0]])
        torque_e = 0
        torque_tilde = -self.K_lat@state_tilde_lat - self.ki_lat*self.integrator_z
        torque = torque_e + torque_tilde

        '''Combine to individual motor forces'''
        F_r = F/2 + (torque/2) / self.d
        F_L = F/2 - (torque/2) / self.d
        
        tau = saturate(np.array([[F_r[0]], [F_L[0]]]), self.F_max)
        
        '''update delayed variables'''
        self.error_h_d1 = error_h
        self.error_z_d1 = error_z
        self.tau_d1 = tau

        return tau, state_hat


    def update_observer(self, y_m):
        # update the observer using RK4 integration
        F1 = self.observer_f(self.state_hat, y_m)
        F2 = self.observer_f(self.state_hat + P.Ts / 2 * F1, y_m)
        F3 = self.observer_f(self.state_hat + P.Ts / 2 * F2, y_m)
        F4 = self.observer_f(self.state_hat + P.Ts * F3, y_m)
        self.state_hat = self.state_hat + P.Ts / 6 * (F1 + 2*F2 + 2*F3 + F4)
        return self.state_hat

    def observer_f(self, state_hat, y_m):
        '''Longitudinal'''
        state_hat_lgtd = np.array([[state_hat[1][0]], [state_hat[4][0]]])
        y_m_lgtd = y_m[1][0]
        h_hat = state_hat_lgtd[0][0]
        F_e = (self.mc + 2*self.mr)*self.g
        # xhatdot = A*(xhat-xe) + B*(u-ue) + L(y-C*xhat)
        statedot_hat_lgtd = self.A_lgtd@state_hat_lgtd \
                           + self.B_lgtd*(np.sum(self.tau_d1) - F_e) \
                              + self.L_lgtd*(y_m_lgtd - self.C_lgtd@state_hat_lgtd)
        h_dot = statedot_hat_lgtd[0][0]
        h_ddot = statedot_hat_lgtd[1][0]
        
        '''Lateral'''
        state_hat_lat = np.array([[state_hat[0][0]], [state_hat[2][0]], [state_hat[3][0]], [state_hat[5][0]]])
        y_m_lat = np.array([[y_m[0][0]], [y_m[2][0]]])
        torque_e = 0.0
        torque = (self.tau_d1[0][0] - self.tau_d1[1][0])*self.d
        # xhatdot = A*(xhat-xe) + B*(u-ue) + L(y-C*xhat)
        statedot_hat_lat = self.A_lat@state_hat_lat \
                          + self.B_lat*(torque-torque_e) \
                            + self.L_lat@(y_m_lat - self.C_lat@state_hat_lat)
        z_dot = statedot_hat_lat[0][0]
        theta_dot = statedot_hat_lat[1][0]
        z_ddot = statedot_hat_lat[2][0]
        theta_ddot = statedot_hat_lat[3][0]

        statedot_hat = np.array([[z_dot],[h_dot],[theta_dot],[z_ddot],[h_ddot],[theta_ddot]])
        return statedot_hat

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


