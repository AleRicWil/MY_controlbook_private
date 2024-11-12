import numpy as np
import control as cnt
import VTOLParam as P

class ctrlStateFeedbackPD:
    def __init__(self, alpha=0.0):
        # Desired responses
            # longitudinal
        tr_h = 1.2 # s, longitudinal rise time
        zeta_h = 1/np.sqrt(2) # longitudinal damping ration
        wn_h = np.pi / (2*tr_h*np.sqrt(1-zeta_h**2)) # longitudinal nat freq
            # lateral
        tr_th = 0.15 # s, lateral inner loop rise time
        zeta_th = 1/np.sqrt(2) # lateral inner damping ratio
        wn_th = np.pi / (2*tr_th*np.sqrt(1-zeta_th**2)) # lateral inner nat freq

        tr_z = 15*tr_th # s, lateral outer loop rise time
        zeta_z = 1/np.sqrt(2) # lateral outer damping ratio
        wn_z = np.pi / (2*tr_z*np.sqrt(1-zeta_z**2)) # lateral outer nat frew
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
        desCharPoly_lgtd = [1, 2 * zeta_h * wn_h, wn_h**2]
        desPoles_lgtd = np.roots(desCharPoly_lgtd)
            # Lateral
        desCharPoly_lat = np.convolve([1, 2 * zeta_z * wn_z, wn_z**2],
                                    [1, 2 * zeta_th * wn_th, wn_th**2])
        desPoles_lat = np.roots(desCharPoly_lat)
        # State Space Equations
            # xdot = A*x + B*u
            #   y  = C*x
        p = self.mc + 2*self.mr
        Fe = p*self.g
            # Longitudinal
        A_lgtd = np.array([[0.0, 1.0],
                           [0.0, 0.0]])

        B_lgtd = np.array([[0.0],
                           [1/p]])

        C_lgtd = np.array([[1.0, 0.0]])
            # Lateral
        A_lat = np.array([[0.0, 0.0    , 1.0       , 0.0],
                          [0.0, 0.0    , 0.0       , 1.0],
                          [0.0, -Fe/p  , -self.mu/p, 0.0],
                          [0.0, 0.0    , 0.0       , 0.0]])
        B_lat = np.array([[0.0],
                          [0.0],
                          [0.0],
                          [1/(self.Jc + 2*self.mr*self.d**2)]])        
        C_lat = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0]])

        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A_lgtd, B_lgtd)) != np.size(A_lgtd,1)  or \
                np.linalg.matrix_rank(cnt.ctrb(A_lat, B_lat)) != np.size(A_lat,1):
            print("The system is not controllable")
        else:
                # Longitudinal
            self.K_lgtd = (cnt.place(A_lgtd, B_lgtd, desPoles_lgtd))
            self.kr_lgtd = -1.0 / (C_lgtd @ np.linalg.inv(A_lgtd - B_lgtd @ self.K_lgtd) @ B_lgtd)
                # Lateral
            self.K_lat = (cnt.acker(A_lat, B_lat, desPoles_lat))
            Cr_lat = np.array([[1.0, 0.0, 0.0, 0.0]])
            self.kr_lat = -1.0 / (Cr_lat @ np.linalg.inv(A_lat - B_lat @ self.K_lat) @ B_lat)
        print('K_lgtd: ', self.K_lgtd)
        print('kr_lgtd: ', self.kr_lgtd)
        print(f'desPoles_lgtd: {desPoles_lgtd}')
        print('K_lat: ', self.K_lat)
        print('kr_lat: ', self.kr_lat)
        print(f'desPoles_lat: {desPoles_lat}')

         # dirty derivative
        self.sigma = 0.01

        self.h_dot = P.hdot0            # estimated derivated of h
        self.z_dot = P.zdot0            # estimated derivative of z
        self.theta_dot = P.thetadot0    # estimated derivative of theta

        self.h_d1 = P.h0                # h delayed by one sample
        self.z_d1 = P.z0                # z delayed by one sample
        self.theta_d1 = P.theta0        # theta delayed by one sample

    def update(self, ref, state):
        z = state[0][0]
        h = state[1][0]
        theta = state[2][0]

        z_ref = ref[0][0]
        h_ref = ref[1][0]

        # Longitudinal
        '''Construct state_lgtd from h sensor'''
            # dirty dertivative
        self.h_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.h_dot \
                     + (2.0 / (2.0*self.sigma + P.Ts)) * ((h - self.h_d1))

        stateSensor_lgtd = np.array([[h], [self.h_dot]])
        
        '''Prep for u = u_e + u_tilde
                    u_tilde = -K*state_tilde + kr*ref_tilde
                                    state_tilde = state - state_e
                                    ref_tilde = ref - y_e'''
        # Eqilibrium point at any value h. Use current h
        state_tilde_lgtd = stateSensor_lgtd - np.array([[h], [0]])
        h_ref_tilde = h_ref - h

        '''PD feedback control for (h_ref - h) to F'''
        F_e = (self.mc + 2*self.mr)*self.g
        F_tilde = -self.K_lgtd@state_tilde_lgtd + self.kr_lgtd*h_ref_tilde
        F = F_e + F_tilde

        # Lateral
        '''Construct state_lat from z and theta sensors'''
            # dirty dertivatives
        self.z_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.z_dot \
                     + (2.0 / (2.0*self.sigma + P.Ts)) * ((z - self.z_d1))
        self.theta_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.theta_dot \
            + (2.0 / (2.0*self.sigma + P.Ts)) * ((theta - self.theta_d1))

        stateSensor_lat = np.array([[z], [theta], [self.z_dot], [self.theta_dot]])

        '''Prep for u = u_e + u_tilde
                    u_tilde = -K*state_tilde + kr*ref_tilde
                                    state_tilde = state - state_e
                                    ref_tilde = ref - y_e'''
        # Eqilibrium point at any value z and theta=0. Use current z
        state_tilde_lat = stateSensor_lat - np.array([[z], [0], [0], [0]])
        z_ref_tilde = z_ref - z

        '''PD feedback control for (z_ref - z) to torque'''
        torque_e = 0
        torque_tilde = -self.K_lat@state_tilde_lat + self.kr_lat*z_ref_tilde
        torque = torque_e + torque_tilde

        '''Combine to individual motor forces'''
        F_r = F/2 + (torque/2) / self.d
        F_L = F/2 - (torque/2) / self.d
        
        tau = saturate(np.array([[F_r[0][0]], [F_L[0][0]]]), self.F_max)
        
        '''update delayed variables'''
        self.h_d1 = h
        self.z_d1 = z
        self.theta_d1 = theta

        return tau

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


