import numpy as np
import control as cnt
import blockbeamParam as P

class ctrlStateFeedback:
    # dirty derivatives to estimate thetadot
    def __init__(self, alpha=0.0):
        #  tuning parameters
        tr_th = 0.2
        zeta_th = 1/np.sqrt(2)
        tr_z = 10*tr_th
        zeta_z = 1/np.sqrt(2)

        self.m1 = P.m1
        self.m2 = P.m2
        self.length = P.length
        self.g = P.g

        # gain calculation
        wn_th = np.pi / (2*tr_th*np.sqrt(1-zeta_th**2))  # natural frequency
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        des_char_poly = np.convolve([1, 2 * zeta_z * wn_z, wn_z**2],
                                    [1, 2 * zeta_th * wn_th, wn_th**2])
        des_poles = np.roots(des_char_poly)
        print(des_char_poly)
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
            # self.K = np.array([[46.52, 5.92, -0.1259, -0.1603]])
            Cr = np.array([[1.0, 0.0, 0.0, 0.0]])
            self.kr = -1.0 / (Cr @ np.linalg.inv(A - B @ self.K) @ B)
        print('K: ', self.K)
        print('kr: ', self.kr)
        print(f'des_poles: {des_poles}')

    def update(self, z_r, state):
        z = state[0][0]
        theta = state[1][0]

        state_tilde = state - np.array([[z], [0], [0], [0]])
        zr_tilde = z_r - z
        # compute Equlibrium linearizing force
        F_e = (self.m1*z + self.m2*self.length/2)*self.g / self.length

        # Compute the state feedback controller
        F_tilde = -self.K @ state_tilde + self.kr * zr_tilde

        # compute total torque
        tau = saturate(F_e + F_tilde[0][0], P.F_max)

        return tau


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u


