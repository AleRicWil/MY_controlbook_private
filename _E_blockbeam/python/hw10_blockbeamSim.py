import numpy as np
import matplotlib.pyplot as plt
import blockbeamParam as P
from signalGenerator import signalGenerator
from blockbeamAnimation import blockbeamAnimation
from dataPlotter import dataPlotter
from blockbeamDynamics import blockbeamDynamics
from blockbeamCtrlPID import blockbeamCtrlPID

# instantiate pendulum, controller, and reference classes
blockbeam = blockbeamDynamics(alpha=0.01)
controller = blockbeamCtrlPID()
z_refSig = signalGenerator(amplitude=0.15, frequency=0.03, y_offset=P.z0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = blockbeamAnimation()

t = P.t_start  # time starts at t_start
y = np.array([[P.z0], [P.theta0], [P.zdot0], [P.thetadot0]])
while t < P.t_end:  # main simulation loop
    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        z_ref = z_refSig.square(t)
        u = controller.update(z_ref, y)
        y = blockbeam.update(u)  # Propagate the dynamics
        t += P.Ts  # advance time by Ts
    # update animation and data plots at rate t_plot
    animation.update(blockbeam.state)
    dataPlot.update(t, blockbeam.state, u, z_ref)
    plt.pause(0.0001)  # allows time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
