import numpy as np
import control as cnt
import blockbeamParam as P

class ctrlStateFeedbackPD:
    def __init__(self, alpha=0.0):
        # Desired responses
        tr_th = 0.2 # s, inner loop rise time
        zeta_th = 1/np.sqrt(2) # inner damping ratio
        wn_th = np.pi / (2*tr_th*np.sqrt(1-zeta_th**2))

        tr_z = 10*tr_th # s, outer loop rise time
        zeta_z = 1/np.sqrt(2) # outer damping ratio
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        # Initialize Self
        self.m1 = P.m1
        self.m2 = P.m2
        self.L = P.length
        self.g = P.g
        self.F_max = P.F_max
        self.Theta_max = P.Theta_max

        # poles of characteristic polynomial
        des_char_poly = np.convolve([1, 2 * zeta_z * wn_z, wn_z**2],
                                    [1, 2 * zeta_th * wn_th, wn_th**2])
        des_poles = np.roots(des_char_poly)
        # State Space Equations
            # xdot = A*x + B*u
            #   y  = C*x
        p = self.m1*P.z0**2 + self.m2*self.L**2 / 3.0
        A = np.array([[0.0                , 0.0    , 1.0, 0.0],
                      [0.0                , 0.0    , 0.0, 1.0],
                      [0.0                , -self.g, 0.0, 0.0],
                      [(-self.m1*self.g)/p, 0.0    , 0.0, 0.0]])
        B = np.array([[0.0          ],
                      [0.0          ],
                      [0.0          ],
                      [self.L/p]])        
        C = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])

        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A, B)) != np.size(A,1):
            print("The system is not controllable")
        else:
            self.K = (cnt.acker(A, B, des_poles))
            Cr = np.array([[1.0, 0.0, 0.0, 0.0]])
            self.kr = -1.0 / (Cr @ np.linalg.inv(A - B @ self.K) @ B)
        print('K: ', self.K)
        print('kr: ', self.kr)
        print(f'des_poles: {des_poles}')

        # dirty derivative
        self.sigma = 0.02

        self.z_dot = P.zdot0            # estimated derivative of z
        self.theta_dot = P.thetadot0    # estimated derivative of theta
        self.z_d1 = P.z0                # z delayed by one sample
        self.theta_d1 = P.theta0        # theta delayed by one sample

    def update(self, z_ref, state):
        z = state[0][0]
        theta = state[1][0]

        '''Construct state from z and theta sensors'''
            # dirty dertivatives
        self.z_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.z_dot \
                     + (2.0 / (2.0*self.sigma + P.Ts)) * ((z - self.z_d1))
        self.theta_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.theta_dot \
            + (2.0 / (2.0*self.sigma + P.Ts)) * ((theta - self.theta_d1))

        stateSensor = np.array([[z], [theta], [self.z_dot], [self.theta_dot]])

        '''Prep for u = u_e + u_tilde
                    u_tilde = -K*state_tilde + kr*ref_tilde
                                    state_tilde = state - state_e
                                    ref_tilde = ref - y_e'''
        # Eqilibrium point at any value z and theta=0. Use current z
        state_tilde = stateSensor - np.array([[z], [0], [0], [0]])
        z_ref_tilde = z_ref - z

        '''PD feedback control for (z_ref - z) to F'''
        F_e = (self.m1*z + self.m2*self.L/2)*self.g / self.L
        F_tilde = -self.K @ state_tilde + self.kr*z_ref_tilde

        tau = saturate(F_e + F_tilde[0][0], P.F_max)

        '''update delayed variables'''
        self.z_d1 = z
        self.theta_d1 = theta

        return tau

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u


