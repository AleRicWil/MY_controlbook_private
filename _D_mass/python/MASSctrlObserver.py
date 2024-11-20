import numpy as np
import control as cnt
import massParam as P

class ctrlObserver:
    def __init__(self):
        '''Full State Feedback'''
        #  Desired Response
        tr_z = 2.0 # s,  rise time
        zeta_z = 1/np.sqrt(2) # damping ratio
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        integrator_pole_z = -2.5
        # Initialize Self
        self.m = P.m
        self.k = P.k
        self.b = P.b
        self.F_max = P.F_max
        # poles of characteristic polynomial
        des_char_poly = np.convolve([1, 2 * zeta_z * wn_z, wn_z**2],
                                    [1, -integrator_pole_z])
        des_poles = np.roots(des_char_poly)
        # State Space Equations
            # xdot = A*x + B*u
            #  y   = C*x
        self.A = np.array([[0.0           , 1.0           ],
                           [-self.k/self.m, -self.b/self.m]])
        self.B = np.array([[0.0       ],
                           [1.0/self.m]])        
        self.C = np.array([[1.0, 0.0]])
        self.Cr = self.C
        # Integral Augmented system matrices
        self.A1 = np.vstack((np.hstack((self.A , np.zeros((len(self.A) , 1)))),
                        np.hstack((-self.Cr, np.zeros((len(self.Cr), 1)))) ))
        self.B1 = np.vstack((self.B, 0.0))

        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(self.A1, self.B1)) != np.size(self.A1,1):
            print("The system is not controllable")
        else:
            self.K1 = cnt.place(self.A1, self.B1, des_poles)
            self.K = self.K1[0,:-1]
            self.ki = self.K1[0][-1]

        '''Observer'''
        tr_obs = tr_z/10
        zeta_obs = 0.6#1/np.sqrt(2)
        wn_obs = np.pi / (2*tr_obs*np.sqrt(1-zeta_obs**2))
        desCharPoly_obs = np.array([1, 2 * zeta_obs * wn_obs, wn_obs**2])
        desPoles_obs = np.roots(desCharPoly_obs)
        
        if np.linalg.matrix_rank(cnt.ctrb(self.A.T, self.C.T)) != np.size(self.C, axis=1):
            print("The system is not observerable")
        else:
            self.L = cnt.acker(self.A.T, self.C.T, desPoles_obs).T

        '''Gains'''
        print('K: ', self.K)
        print('ki: ', self.ki)
        print(f'desPoles: {des_poles}')
        print('L.T: ', self.L.T)
        print(f'desPoles_obs: {desPoles_obs}')
        

        # integrator, previous input, estimated state
        self.error_z_d1 = 0.0   # z error delayed by one sample
        self.integrator_z = 0.0 # z integrator
        self.tau_d1 = 0.0
        self.state_hat = np.array([[P.z0], [P.zdot0]])


    def update(self, z_ref, y_measured):
        state_hat = self.update_observer(y_measured)
        z_hat = state_hat[0][0]
        zdot_hat = state_hat[1][0]
        
        error_z = z_ref - z_hat
        self.integrator_z = self.integrator_z + (P.Ts / 2) * (error_z + self.error_z_d1)
        # print(f'z: ', self.integrator_z)

        '''Feedback control for (z_ref - z) to F'''
        F_e = self.k*z_hat
        F_tilde = -self.K @ state_hat - self.ki*self.integrator_z
        tau = saturate(F_e + F_tilde[0], P.F_max)

        '''update delayed variables'''
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
        # compute feedback linearizing torque tau_fl
        z_hat = state_hat[0][0]
        F_e = self.k*z_hat 
        # xhatdot = A*(xhat-xe) + B*(u-ue) + L(y-C*xhat)
        statedot_hat = self.A@state_hat + self.B*(self.tau_d1-F_e) + self.L*(y_m[0] - self.C@state_hat)
        return statedot_hat



def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u

