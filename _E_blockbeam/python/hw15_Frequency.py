import blockbeamParam as P
import numpy as np
import control
import matplotlib.pyplot as plt

z_e = P.length/2
Plant = control.tf([P.length/(P.m1*z_e**2 + P.m2*P.length**2/3)],
                   [1, 0.0, 0.0])
Plant2 = control.tf([-P.g],
                    [1, 0.0, 0.0])

omega = np.logspace(-3, 3, 100)

fig = plt.figure(1)
control.bode(Plant, dB=True, display_margins=False)
fig.axes[0].set_title('P(s) for force to theta')
for ax in fig.axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

fig = plt.figure(2)
control.bode(Plant2, dB=False, display_margins=False)
fig.axes[0].set_title('P(s) for theta to z')
for ax in fig.axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

# mag, phase, omega = control.frequency_response(Plant, omega=[0.3, 10.0, 1000.0])

# print(mag, phase, omega)
plt.show()