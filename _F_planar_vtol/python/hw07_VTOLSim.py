import numpy as np
import matplotlib.pyplot as plt
import VTOLParam as P
from signalGenerator import signalGenerator
from VTOLAnimation import VTOLAnimation
from dataPlotter import dataPlotter
from VTOLDynamics import VTOLDynamics
from VTOLCtrlPD import VTOLCtrlPD

# instantiate VTOL, controller, and reference classes
VTOL = VTOLDynamics(alpha=0.0)
controller = VTOLCtrlPD()
z_refSig = signalGenerator(amplitude=1, frequency=0.03, y_offset=P.z0)
h_refSig = signalGenerator(amplitude=1, frequency=0.1, y_offset=P.h0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = VTOLAnimation()

# initial conditions
y = np.array([[P.z0], [P.h0], [P.theta0], [P.zdot0], [P.hdot0], [P.thetadot0]], dtype=np.float64)
t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        z_ref = z_refSig.square(t)
        h_ref = P.z0#h_refSig.square(t)
        u = controller.update([[z_ref], [h_ref]], y)
        y = VTOL.update(u)  # Propagate the dynamics
        t += P.Ts  # advance time by Ts
    # update animation and data plots at rate t_plot
    animation.update(VTOL.state)
    dataPlot.update(t, VTOL.state, u, z_ref, h_ref)
    plt.pause(0.01)  # allows time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
