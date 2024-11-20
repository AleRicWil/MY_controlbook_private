import numpy as np
import control as cnt
import blockbeamParam as P
from scipy import signal

class ctrlObserver:
    def __init__(self):
        self.count = 0
        '''Full State Feedback'''
        # Desired responses
        tr_th = 0.2 # s, inner loop rise time
        zeta_th = 1/np.sqrt(2) # inner damping ratio
        wn_th = np.pi / (2*tr_th*np.sqrt(1-zeta_th**2))

        tr_z = 10*tr_th # s, outer loop rise time
        zeta_z = 1/np.sqrt(1.2) # outer damping ratio
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        integrator_pole_z = -10.0
        # Initialize Self
        self.m1 = P.m1
        self.m2 = P.m2
        self.L = P.length
        self.g = P.g
        self.F_max = P.F_max
        self.Theta_max = P.Theta_max
        # poles of characteristic polynomial
        TEMP_des_char_poly = np.convolve([1, 2 * zeta_z * wn_z, wn_z**2],
                                         [1, 2 * zeta_th * wn_th, wn_th**2])
        des_char_poly = np.convolve(TEMP_des_char_poly,
                                    [1, -integrator_pole_z])
        des_poles = np.roots(des_char_poly)
        # State Space Equations
            # xdot = A*x + B*u
            #   y  = C*x
        p = self.m1*P.z0**2 + self.m2*self.L**2 / 3.0
        self.A = np.array([[0.0                , 0.0    , 1.0, 0.0],
                      [0.0                , 0.0    , 0.0, 1.0],
                      [0.0                , -self.g, 0.0, 0.0],
                      [(-self.m1*self.g)/p, 0.0    , 0.0, 0.0]])
        self.B = np.array([[0.0          ],
                      [0.0          ],
                      [0.0          ],
                      [self.L/p]])        
        self.C = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])
        self.Cr = np.array([[1.0, 0.0, 0.0, 0.0]])
        # Integral Augmented system matrices
        A1 = np.vstack((np.hstack((self.A , np.zeros((len(self.A) , 1)))),
                        np.hstack((-self.Cr, np.zeros((len(self.Cr), 1)))) ))
        B1 = np.vstack((self.B, 0.0))
        
        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A1, B1)) != np.size(A1,1):
            print("The system is not controllable")
        else:
            self.K1 = (cnt.acker(A1, B1, des_poles))
            self.K = self.K1[0,:-1]
            self.ki = self.K1[0][-1]

        '''Observer'''
        tr_z_obs = tr_z/20
        zeta_z_obs = 1/np.sqrt(2)
        wn_z_obs = np.pi / (2*tr_z_obs*np.sqrt(1-zeta_z_obs**2))
        
        tr_th_obs = tr_th/20
        zeta_th_obs = 1/np.sqrt(2)
        wn_th_obs = np.pi / (2*tr_th_obs*np.sqrt(1-zeta_th_obs**2))
        
        desCharPoly_obs = np.convolve([1, 2 * zeta_z_obs * wn_z_obs, wn_z_obs**2],
                                      [1, 2 * zeta_th_obs * wn_th_obs, wn_th_obs**2])
        desPoles_obs = np.roots(desCharPoly_obs)
        # Compute the observer gains if the system is observable
        if np.linalg.matrix_rank(cnt.ctrb(self.A.T, self.C.T)) != np.size(self.C, axis=1):
            print("The system is not observable")
        else:
            self.L_mat = signal.place_poles(self.A.T, self.C.T, desPoles_obs).gain_matrix.T


        print('K: ', self.K)
        print('ki: ', self.ki)
        print(f'des_poles: {des_poles}')
        print('L.T: ', self.L_mat.T)
        print(f'desPoles_obs: {desPoles_obs}')
        
        
        self.error_z_d1 = 0.0           # z error delayed by one sample
        self.integrator_z = 0.0         # z integrator

        self.state_hat = np.array([[P.z0], [0.0], [0.0], [0.0]])
        self.tau_d1 = (self.m1*P.z0 + self.m2*self.L/2)*self.g / self.L

    def update(self, z_ref, y_measured):
        state_hat = self.update_observer(y_measured)
        z_hat = state_hat[0][0]

        error_z = z_ref - z_hat
        self.integrator_z = self.integrator_z + (P.Ts / 2) * (error_z + self.error_z_d1)
    

        '''PID feedback control for (z_ref - z) to F'''
        F_e = (self.m1*z_hat + self.m2*self.L/2)*self.g / self.L
        F_tilde = -self.K @ state_hat - self.ki*self.integrator_z

        tau = saturate(F_e + F_tilde[0], P.F_max)

        '''update delayed variables'''

        self.error_z_d1 = error_z
        self.tau_d1 = tau

        self.count += 1
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
        # compute feedback linearizing torque tau_fl
        z_hat = state_hat[0][0]
        F_e = (self.m1*z_hat + self.m2*self.L/2)*self.g / self.L
        # xhatdot = A*(xhat-xe) + B*(u-ue) + L(y-C*xhat)
        statedot_hat = self.A@state_hat + self.B*(self.tau_d1-F_e) + self.L_mat@(y_m - self.C@state_hat)
        return statedot_hat

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u
