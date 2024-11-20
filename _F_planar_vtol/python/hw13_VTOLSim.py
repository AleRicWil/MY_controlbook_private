import numpy as np
import matplotlib.pyplot as plt
import VTOLParam as P
from signalGenerator import signalGenerator
from VTOLAnimation import VTOLAnimation
from dataPlotter import dataPlotter
from VTOLDynamics import VTOLDynamics
from VTOLctrlObserver import ctrlObserver
from dataPlotterObserver import dataPlotterObserver

# instantiate VTOL, controller, and reference classes
VTOL = VTOLDynamics(alpha=0.5)
controller = ctrlObserver()
z_refSig = signalGenerator(amplitude=2, frequency=0.05, y_offset=P.z0)
h_refSig = signalGenerator(amplitude=1, frequency=0.03, y_offset=P.h0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
dataPlotObserver = dataPlotterObserver()
animation = VTOLAnimation()

# initial conditions
y = np.array([[P.z0], [P.h0], [P.theta0]])#, [P.zdot0], [P.hdot0], [P.thetadot0]], dtype=np.float64)
t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        z_ref = z_refSig.square(t)
        h_ref = P.h0#h_refSig.square(t)
        u, state_hat = controller.update([[z_ref], [h_ref]], y)
        y = VTOL.update(u)  # Propagate the dynamics
        t += P.Ts  # advance time by Ts
    # update animation and data plots at rate t_plot
    animation.update(VTOL.state)
    dataPlot.update(t, VTOL.state, u, z_ref, h_ref)
    state_hat_lat = np.array([[state_hat[0][0]], [state_hat[2][0]], [state_hat[3][0]], [state_hat[5][0]]])
    state_hat_lgtd = np.array([[state_hat[1][0]], [state_hat[4][0]]])
    dataPlotObserver.update(t=t, x=VTOL.state, xhat_lat=state_hat, xhat_lon=state_hat_lgtd)
    plt.pause(0.01)  # allows time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
