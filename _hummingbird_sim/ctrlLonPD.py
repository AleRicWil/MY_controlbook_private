import numpy as np
import hummingbirdParam as P


class ctrlLonPD:
    def __init__(self):
        # tuning parameters
        tr_theta = 1
        zeta_theta = 0.707
        # gain calculation
        b_theta = P.ellT/(P.m1*P.ell1**2 + P.m2*P.ell2**2 + P.J1y + P.J2y)
        #print('b_theta: ', b_theta)
        wn_theta = 2.2 / tr_theta
        self.kp_theta = wn_theta**2 / b_theta
        self.kd_theta = (2*zeta_theta*wn_theta) / b_theta
        # print gains to terminal
        print('kp_theta: ', self.kp_theta)
        print('kd_theta: ', self.kd_theta) 
        # sample rate of the controller
        self.Ts = P.Ts
        # dirty derivative parameters
        self.sigma = 0.05  # cutoff freq for dirty derivative
        # delayed variables
        self.theta_d1 = 0.
        self.theta_dot = 0.
        self.error_theta_d1 = 0.  # pitch error delayed by 1

    def update(self, theta_ref, state: np.ndarray):
        theta_r = theta_ref
        theta = state[1][0]
        force_fl = (P.m1*P.ell1 + P.m2*P.ell2)*P.g*np.cos(theta)/P.ellT
        # compute errors
        error_theta = theta_r - theta
        # update differentiator
        self.theta_dot = (2.0*self.sigma - P.Ts) / (2.0*self.sigma + P.Ts) * self.theta_dot \
                         + (2.0 / (2.0*self.sigma + P.Ts)) * ((theta - self.theta_d1))
        
        # pitch control
        force_tilde = self.kp_theta*error_theta - self.kd_theta*self.theta_dot
        force_unsat = force_fl + force_tilde
        force = saturate(force_unsat, -P.force_max, P.force_max)
        torque = 0.
        # convert force and torque to pwm signals
        tau = np.array([[force + (torque / P.d)],   # u_left
                        [force - (torque / P.d)]])  # u_right          
        pwm = tau / (2 * P.km)
        pwm = saturate(pwm, 0, 1)
        # update all delayed variables
        self.theta_d1 = theta
        self.error_theta_d1 = error_theta
        # return pwm plus reference signals
        return pwm#, np.array([[0.], [theta_r], [0.]])


def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        if u > up_limit:
            u = up_limit
        if u < low_limit:
            u = low_limit
    else:
        for i in range(0, u.shape[0]):
            if u[i][0] > up_limit:
                u[i][0] = up_limit
            if u[i][0] < low_limit:
                u[i][0] = low_limit
    return u




