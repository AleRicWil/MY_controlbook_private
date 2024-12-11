import VTOLParam as P
import numpy as np
import control
import matplotlib.pyplot as plt
from VTOLCtrlPID import VTOLCtrlPID

PID = VTOLCtrlPID()

'''Longitudinal'''
Plant = control.tf([1/(P.mc + 2*P.mr)],
                   [1, 0.0, 0.0])
C_lgt_pid = control.tf([(PID.kd_h+PID.kp_h*PID.sigma), 
                    (PID.kp_h+PID.ki_h*PID.sigma), 
                     PID.ki_h],
                   [PID.sigma, 1, 0])
ref = control.tf([1],
                 [1,0,0,0])
# print(Plant*C_lgt_pid)

# fig = plt.figure(1)
# control.bode([Plant, Plant*C_lgt_pid, ref], omega_limits=[0.01, 1000], dB=True, display_margins=False)
# fig.axes[0].set_title('P(s) for force to h')
# for ax in fig.axes:
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

# mag, phase, omega = control.frequency_response(Plant*C_lgt_pid, omega=[30])

# # Noise
# Bn = -20 * np.log10(abs(mag[omega==30][0]))
# print('Bn', Bn)
# print('gam_n', 10**(-Bn/20)*100)

'''Lateral'''
# Inner loop
Plant2 = control.tf([1/(P.Jc + 2*P.mr*P.d**2)],
                    [1, 0.0, 0.0])
C_ltl_pd = control.tf([PID.kd_th + PID.sigma * PID.kp_th, PID.kp_th], 
                      [PID.sigma, 1])

# fig = plt.figure(2)
# control.bode([Plant2, Plant2*C_ltl_pd], dB=True, omega_limits=[0.1, 1000], display_margins=False)
# fig.axes[0].set_title('P(s) for torque to theta, inner loop')
# for ax in fig.axes:
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

# mag, phase, omega = control.frequency_response(Plant2*C_ltl_pd, omega=[2.0])
# mag0, phase0, omega0 = control.frequency_response(Plant2, omega=[2.0])

# # Disturbance input
# Bdi = 20 * np.log10(abs(mag[omega==2.0][0])) - 20 * np.log10(abs(mag0[omega==2.0][0]))
# print('Bdi', Bdi)
# print('gam_di', 10**(-Bdi/20)*100)


# Outer loop
F_e = (P.mc + 2*P.mr)*P.g
Plant3 = control.tf([-F_e],
                    [P.mc + 2*P.mr, P.mu, 0.0])
C_ltl_pid = control.tf([(PID.kd_z+PID.kp_z*PID.sigma), 
                        (PID.kp_z+PID.ki_z*PID.sigma), 
                         PID.ki_z],
                       [PID.sigma, 1, 0])

fig = plt.figure(3)
control.bode([Plant3, Plant3*C_ltl_pid], dB=True, omega_limits=[0.001, 1000], display_margins=False)
fig.axes[0].set_title('P(s) for theta to z, outer loop')
for ax in fig.axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

mag, phase, omega = control.frequency_response(Plant, omega=[0.01])

# Reference
# Br = 20 * np.log10(abs(mag[omega==0.1][0]))
# print('\nBr', Br)
# print('gam_r', 10**(-Br/20)*100)

# Disturbance output
Bdo = 20 * np.log10(abs(mag[omega==0.01][0]))
print('Bdo', Bdo)
print('gam_do', 10**(-Bdo/20)*100)

# print(mag, phase, omega)
plt.show()