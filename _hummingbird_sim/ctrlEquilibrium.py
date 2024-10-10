import numpy as np
import hummingbirdParam as P
from hummingbirdDynamics import HummingbirdDynamics as HD

class ctrlEquilibrium:
    def __init__(self):
        pass 

    def update(self, x):
        theta = x[0][0]
        thetadot = x[1][0]
        
        force_equilibrium = (P.m1*P.ell1 + P.m2*P.ell2)*P.g / P.ellT
        force = force_equilibrium
        torque_equilibrium = 0.0
        torque = torque_equilibrium
        # convert force and torque to pwm signals
        uL = (force + torque/P.d) / (2*P.km)
        uR = (force - torque/P.d) / (2*P.km)
        pwm = np.array([[uL], [uR]])
        pwm = saturate(pwm, 0, 1) 
        return pwm
    
    def update2(self, x, y):
        theta_ref = x[0][0]
        thetadot_ref = x[1][0]

        phi = y[0][0]
        theta = y[1][0]
        print(theta*180/np.pi)
        psi = y[2][0]

        force_equilibrium = (P.m1*P.ell1 + P.m2*P.ell2)*P.g / P.ellT
        force = force_equilibrium   
        if theta > theta_ref:
            force -= force_equilibrium*(abs(theta))*0.1
        elif theta < theta_ref:
            force += force_equilibrium*(abs(theta))*0.1
        
        torque_equilibrium = 0.0
        torque = torque_equilibrium
        # convert force and torque to pwm signals
        uL = (force + torque/P.d) / (2*P.km)
        uR = (force - torque/P.d) / (2*P.km)
        pwm = np.array([[uL], [uR]])
        pwm = saturate(pwm, 0, 1) 
        return pwm


def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        u = np.max((np.min((u, up_limit)), low_limit))
    else:
        for i in range(0, u.shape[0]):
            u[i][0] = np.max((np.min((u[i][0], up_limit)), low_limit))
    return u




