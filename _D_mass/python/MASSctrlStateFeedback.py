import numpy as np
import control as cnt
import massParam as P

class ctrlStateFeedback:
    # dirty derivatives to estimate thetadot
    def __init__(self, alpha=0.0):
        #  tuning parameters
        tr = 2.0
        zeta = 0.707

        self.m = P.m
        self.k = P.k
        self.b = P.b

        # gain calculation
        wn = 2.2 / tr  # natural frequency
        des_char_poly = [1, 2 * zeta * wn, wn**2]
        des_poles = np.roots(des_char_poly)
        #des_poles = [-50.0, -50.1]

        # State Space Equations
        # xdot = A*x + B*u
        # y = C*x

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

    def update(self, z_r, state):
        z = state[0][0]
        # compute equilibrium force
        F_e = 0

        # Compute the state feedback controller
        F_tilde = -self.K @ state + self.kr * z_r

        # compute total torque
        tau = saturate(F_e + F_tilde[0][0], P.F_max)

        return tau


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u


