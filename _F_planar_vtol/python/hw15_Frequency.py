import VTOLParam as P
import numpy as np
import control
import matplotlib.pyplot as plt

z_e = P.length/2
Plant = control.tf([1/(P.mc + 2*P.mr)],
                   [1, 0.0, 0.0])
Plant2 = control.tf([1/(P.Jc + 2*P.mr*P.d**2)],
                    [1, 0.0, 0.0])
F_e = (P.mc + 2*P.mr)*P.g
Plant3 = control.tf([-F_e],
                    [P.mc + 2*P.mr, P.mu, 0.0])

omega = np.logspace(-2, 1, 100)

fig = plt.figure(1)
control.bode(Plant, dB=True, display_margins=False)
fig.axes[0].set_title('P(s) for force to h')
for ax in fig.axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

fig = plt.figure(2)
control.bode(Plant2, dB=True, display_margins=False)
fig.axes[0].set_title('P(s) for torque to theta')
for ax in fig.axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

fig = plt.figure(3)
control.bode(Plant3, dB=True, omega=omega, display_margins=False)
fig.axes[0].set_title('P(s) for theta to z')
for ax in fig.axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

# mag, phase, omega = control.frequency_response(Plant, omega=[0.3, 10.0, 1000.0])

# print(mag, phase, omega)
plt.show()