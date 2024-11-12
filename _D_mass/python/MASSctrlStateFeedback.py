import numpy as np
import control as cnt
import massParam as P

class ctrlStateFeedbackPD:
    def __init__(self, alpha=0.0):
        #  Desired Response
        tr_z = 2.0 # s, inner loop rise time
        zeta_z = 1/np.sqrt(2) # inner damping ratio
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        # Initialize Self
        self.m = P.m
        self.k = P.k
        self.b = P.b
        self.F_max = P.F_max
        # poles of characteristic polynomial
        des_char_poly = [1, 2 * zeta_z * wn_z, wn_z**2]
        des_poles = np.roots(des_char_poly)
        # State Space Equations
            # xdot = A*x + B*u
            #  y   = C*x
        A = np.array([[0.0           , 1.0           ],
                      [-self.k/self.m, -self.b/self.m]])
        B = np.array([[0.0       ],
                      [1.0/self.m]])        
        C = np.array([[1.0, 0.0]])

        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A, B)) != np.size(A,1):
            print("The system is not controllable")
        else:
            self.K = (cnt.place(A, B, des_poles))
            self.kr = -1.0 / (C @ np.linalg.inv(A - B @ self.K) @ B)
        print('K: ', self.K)
        print('kr: ', self.kr)
        print(f'des_poles: {des_poles}')

        # dirty derivative
        self.sigma = 0.02
        self.z_dot = P.zdot0    # estimated derivative of z
        self.z_d1 = P.z0        # z delayed by one sample

    def update(self, z_ref, state):
        z = state[0][0]

        '''Construct state from z sensor'''
            # dirty dertivative
        self.z_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.z_dot \
                     + (2.0 / (2.0*self.sigma + P.Ts)) * ((z - self.z_d1))

        stateSensor = np.array([[z], [self.z_dot]])
        
        '''Prep for u = u_e + u_tilde
                    u_tilde = -K*state_tilde + kr*ref_tilde
                                    state_tilde = state - state_e
                                    ref_tilde = ref - y_e'''
        # Eqilibrium point at any value z. Use current z
        state_tilde = stateSensor - np.array([[z], [0]])
        z_ref_tilde = z_ref - z

        '''PD feedback control for (z_ref - z) to F'''
        F_e = self.k*z
        F_tilde = -self.K@state_tilde + self.kr*z_ref_tilde

        tau = saturate(F_e + F_tilde[0][0], P.F_max)

        '''update delayed variables'''
        self.z_d1 = z

        return tau


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u


