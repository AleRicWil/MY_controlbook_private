import numpy as np
import control as cnt
import blockbeamParam as P

class ctrlStateFeedback:
    # dirty derivatives to estimate thetadot
    def __init__(self, alpha=0.0):
        #  tuning parameters
        tr_th = 1.0
        zeta_th = 0.707
        tr_z = 10*tr_th
        zeta_z = 0.707

        self.m1 = P.m1
        self.m2 = P.m2
        self.length = P.length
        self.g = P.g

        # gain calculation
        wn_th = 2.2 / tr_th  # natural frequency
        wn_z = 2.2 / tr_z
        des_char_poly = np.convolve([1, 2 * zeta_th * wn_th, wn_th**2],
                                    [1, 2 * zeta_z * wn_z, wn_z**2])
        des_poles = np.roots(des_char_poly)
        #des_poles = [-50.0, -50.1]

        # State Space Equations
        # xdot = A*x + B*u
        # y = C*x
        p = self.m1*P.z0**2 + self.m2*self.length**2 / 3.0
        A = np.array([[0.0                , 0.0    , 1.0, 0.0],
                      [0.0                , 0.0    , 0.0, 1.0],
                      [0.0                , -self.g, 0.0, 0.0],
                      [(-self.m1*self.g)/p, 0.0    , 0.0, 0.0]])
        B = np.array([[0.0          ],
                      [0.0          ],
                      [0.0          ],
                      [self.length/p]])        
        C = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])

        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A, B)) != np.size(A,1):
            print("The system is not controllable")
        else:
            self.K = (cnt.acker(A, B, des_poles))
            self.K = np.array([[7.44, 2.37, -0.0201, -0.0641]])
            Cr = np.array([[1.0, 0.0, 0.0, 0.0]])
            self.kr = -1.0 / (Cr @ np.linalg.inv(A - B @ self.K) @ B)
        print('K: ', self.K)
        print('kr: ', self.kr)
        print(f'des_poles: {des_poles}')

    def update(self, z_r, state):
        z = state[0][0]
        theta = state[1][0]
        # compute Equlibrium linearizing force
        F_e = (self.m1*z + self.m2*self.length/2)*self.g / self.length

        # Compute the state feedback controller
        F_tilde = -self.K @ state + self.kr * z_r

        # compute total torque
        tau = saturate(F_e + F_tilde[0][0], P.F_max)

        return tau


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u


