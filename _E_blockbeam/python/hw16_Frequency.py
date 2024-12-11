import blockbeamParam as P
import numpy as np
import control
import matplotlib.pyplot as plt
from blockbeamCtrlPID import blockbeamCtrlPID

PID = blockbeamCtrlPID()

'''Inner Loop'''
z_e = P.length/2
Plant = control.tf([P.length/(P.m1*z_e**2 + P.m2*P.length**2/3)],
                   [1, 0.0, 0.0])
C_pd = control.tf([PID.kd_th + PID.sigma * PID.kp_th, PID.kp_th], 
                  [PID.sigma, 1])

# fig = plt.figure(1)
# control.bode([Plant, Plant*C_pd], dB=True, omega_limits=[0.1, 1000], display_margins=False)
# fig.axes[0].set_title('P(s) for force to theta')
# for ax in fig.axes:
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

# mag, phase, omega = control.frequency_response(Plant*C_pd, omega=[0.1])
# mag0, phase0, omega0 = control.frequency_response(Plant, omega=[0.8])

# Reference
# Br = 20 * np.log10(abs(mag[omega==1.0][0]))
# print('\nBr', Br)
# print('gam_r', 10**(-Br/20)*100)

# Disturbance input
# Bdi = 20 * np.log10(abs(mag[omega==0.8][0])) - 20 * np.log10(abs(mag0[omega==0.8][0]))
# print('Bdi', Bdi)
# print('gam_di', 10**(-Bdi/20)*100)

# Noise
# Bn = -20 * np.log10(abs(mag[omega==300][0]))
# print('Bn', Bn)
# print('gam_n', 10**(-Bn/20)*100)

'''Outer Loop'''
Plant2 = control.tf([-P.g],
                    [1, 0.0, 0.0])
C_pid = control.tf([(PID.kd_z+PID.kp_z*PID.sigma), 
                    (PID.kp_z+PID.ki_z*PID.sigma), 
                     PID.ki_z],
                   [PID.sigma, 1, 0])
ref = control.tf([1],
                 [1,0])


fig = plt.figure(2)
control.bode([Plant2, Plant2*C_pid, ref], dB=True, omega_limits=[0.01, 1000], display_margins=False)
fig.axes[0].set_title('P(s) for theta to z')
for ax in fig.axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

mag, phase, omega = control.frequency_response(Plant2*C_pid, omega=[0.6])

# Disturbance output
# Bdo = 20 * np.log10(abs(mag[omega==0.1][0]))
# print('Bdo', Bdo)
# print('gam_do', 10**(-Bdo/20)*100)

# Reference
Br = 20 * np.log10(abs(mag[omega==0.6][0]))
print('\nBr', Br)
print('gam_r', 10**(-Br/20))



plt.show()