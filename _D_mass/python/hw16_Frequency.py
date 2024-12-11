import massParam as P
import numpy as np
import control
import matplotlib.pyplot as plt
from massCtrlPID import massCtrlPID

PID = massCtrlPID()

ref = control.tf([1],
                 [1,0])

Plant = control.tf([1/P.m],
                   [1, P.b/P.m, P.k/P.m])
C_pid = control.tf([(PID.kd_z+PID.kp_z*PID.sigma), 
                    (PID.kp_z+PID.ki_z*PID.sigma), 
                     PID.ki_z],
                   [PID.sigma, 1, 0])

omega = np.logspace(-1, 2, 1000)

fig = plt.figure()
control.bode([Plant, Plant*C_pid, ref], dB=True, omega=omega, display_margins=False)
fig.axes[0].set_title('P(s) for spring-mass-damper')
for ax in fig.axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

mag0, phase0, omega0 = control.frequency_response(Plant, omega=[0.001, 100])
mag, phase, omega = control.frequency_response(Plant*C_pid, omega=[0.001, 100])
mag2, phase2, omega2 = control.frequency_response(ref, omega=[0.001, 100])
# Unit ramp input
A = 1.0
B1 = 20 * np.log10(abs(mag[omega==0.001][0])) - 20 * np.log10(abs(mag2[omega==0.001][0]))
Mv = 10**(B1/20)
print('B1', B1)
print('Mv', Mv)
print('A/Mv', A/Mv)
# print(omega)

# Disturbance input
# Bdi = 20 * np.log10(abs(mag[omega==0.1][0])) - 20 * np.log10(abs(mag0[omega==0.1][0]))
# print('Bdi', Bdi)
# print('gam_di', 10**(-Bdi/20))

# Noise
Bn = -20 * np.log10(abs(mag[omega==100][0]))
print('Bn', Bn)
print('gam_n', 10**(-Bn/20)*100)


print(Plant*C_pid)

plt.show()