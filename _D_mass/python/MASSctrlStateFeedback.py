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

        '''PID feedback control for (z_ref - z) to F'''
        F_e = self.k*z
        F_tilde = -self.K@state_tilde + self.kr*z_ref_tilde
        print(state_tilde)
        tau = saturate(F_e + F_tilde[0][0], P.F_max)

        '''update delayed variables'''
        self.z_d1 = z

        return tau

class ctrlStateFeedbackPID:
    def __init__(self, alpha=0.0):
        #  Desired Response
        tr_z = 2.0 # s,  rise time
        zeta_z = 1/np.sqrt(2) # damping ratio
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2))
        integrator_pole_z = -0.3
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
        A = np.array([[0.0           , 1.0           ],
                      [-self.k/self.m, -self.b/self.m]])
        B = np.array([[0.0       ],
                      [1.0/self.m]])        
        C = np.array([[1.0, 0.0]])
        Cr = C
        # Integral Augmented system matrices
        A1 = np.vstack((np.hstack((A , np.zeros((len(A) , 1)))),
                        np.hstack((-Cr, np.zeros((len(Cr), 1)))) ))
        B1 = np.vstack((B, 0.0))

        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A1, B1)) != np.size(A1,1):
            print("The system is not controllable")
        else:
            self.K1 = cnt.place(A1, B1, des_poles)
            self.K = self.K1[0,:-1]
            self.ki = self.K1[0][-1]
        print('K: ', self.K)
        print('ki: ', self.ki)
        print(f'des_poles: {des_poles}')
        self.zdot_min = 0.1     # anti-windup threshold

        # dirty derivative and integrator
        self.sigma = 0.02

        self.z_dot = P.zdot0    # estimated derivative of z
        self.z_d1 = P.z0        # z delayed by one sample
        self.error_z_d1 = 0.0   # z error delayed by one sample
        self.integrator_z = 0.0 # z integrator

    def update(self, z_ref, state):
        z = state[0][0]

        '''Construct state from z sensor'''
            # dirty dertivative
        self.z_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.z_dot \
                     + (2.0 / (2.0*self.sigma + P.Ts)) * ((z - self.z_d1))

        stateSensor = np.array([[z], [self.z_dot]])
        
        '''Prep for u = u_e + u_tilde
                    u_tilde = -K*state_tilde - ki*integrator_error_refVar
                                    state_tilde = state - state_e
                                    error_refVar = ref - y_ref'''
        # Eqilibrium point at any value z. Use current z_ref
        state_tilde = stateSensor - np.array([[z_ref], [0]])
            # anti-windup integral
        error_z = z_ref - z
        if abs(self.z_dot) < self.zdot_min:
            self.integrator_z = self.integrator_z + (P.Ts / 2) * (error_z + self.error_z_d1)
            print(f'z: ', self.integrator_z)
        else:
            print('.')
            pass

        '''PD feedback control for (z_ref - z) to F'''
        F_e = self.k*z_ref
        F_tilde = -self.K @ state_tilde - self.ki*self.integrator_z
        # print(state_tilde)
        tau = saturate(F_e + F_tilde[0], P.F_max) + 0.25

        '''update delayed variables'''
        self.error_z_d1 = error_z
        self.z_d1 = z

        return tau



def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u


