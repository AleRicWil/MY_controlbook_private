import numpy as np
import matplotlib.pyplot as plt
import VTOLParam as P
from signalGenerator import signalGenerator
from VTOLAnimation import VTOLAnimation
from dataPlotter import dataPlotter
from VTOLDynamics import VTOLDynamics

hover = 7.35

# instantiate VTOL, controller, and reference classes
VTOL = VTOLDynamics(alpha=0.0)
F_L = signalGenerator(amplitude=1, frequency=0.5, y_offset=hover)
F_r = signalGenerator(amplitude=1, frequency=0.5, y_offset=hover)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = VTOLAnimation()

u = np.array([[0], [0]], dtype=np.float64)
t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        u[0][0] = F_r.sin(t)
        u[1][0] = F_L.sin(t)
        y = VTOL.update(u)  # Propagate the dynamics
        t += P.Ts  # advance time by Ts
    # update animation and data plots at rate t_plot
    animation.update(VTOL.state)
    dataPlot.update(t, VTOL.state, u)
    plt.pause(0.01)  # allows time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
