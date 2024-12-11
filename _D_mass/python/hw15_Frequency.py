import massParam as P
import numpy as np
import control
import matplotlib.pyplot as plt

dB_flag = False

th_e = 0

Plant = control.tf([1/P.m],
                   [1, P.b/P.m, P.k/P.m])

omega = np.logspace(-2, 1, 100)

fig = plt.figure()
control.bode(Plant, dB=True, omega=omega, display_margins=False)
fig.axes[0].set_title('P(s) for spring-mass-damper')
for ax in fig.axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
mag, phase, omega = control.frequency_response(Plant, omega=[0.3, 10.0, 1000.0])

print(mag, phase, omega)
plt.show()