import numpy as np
import matplotlib.pyplot as plt
import blockbeamParam as P
from signalGenerator import signalGenerator
from blockbeamAnimation import blockbeamAnimation
from dataPlotter import dataPlotter
from blockbeamDynamics import blockbeamDynamics
from BLOCKBEAMctrlObserver import ctrlObserver
from dataPlotterObserver import dataPlotterObserver

# instantiate pendulum, controller, and reference classes
blockbeam = blockbeamDynamics(alpha=0.0)
controller = ctrlObserver()
z_refSig = signalGenerator(amplitude=0.15, frequency=0.05, y_offset=P.z0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
dataPlotObserver = dataPlotterObserver()
animation = blockbeamAnimation()


y = np.array([[P.z0], [P.theta0]])
t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        z_ref = z_refSig.square(t)
        u, state_hat = controller.update(z_ref, y)
        y = blockbeam.update(u)  # Propagate the dynamics
        t += P.Ts  # advance time by Ts
    # update animation and data plots at rate t_plot
    animation.update(blockbeam.state)
    dataPlot.update(t, blockbeam.state, u, z_ref)
    dataPlotObserver.update(t=t, x=blockbeam.state, x_hat=state_hat, d=0.0, d_hat=0.0)
    plt.pause(0.0001)  # allows time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
