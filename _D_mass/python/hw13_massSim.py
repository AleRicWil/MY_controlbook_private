import numpy as np
import matplotlib.pyplot as plt
import massParam as P
from signalGenerator import signalGenerator
from massAnimation import massAnimation
from dataPlotter import dataPlotter
from massDynamics import massDynamics
from MASSctrlObserver import ctrlObserver
from dataPlotterObserver import dataPlotterObserver

# instantiate arm, controller, and reference classes
mass = massDynamics(alpha=0.0)
controller = ctrlObserver()
z_refSig = signalGenerator(amplitude=1, frequency=0.03)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
dataPlotObserver = dataPlotterObserver()
animation = massAnimation()

t = P.t_start  # time starts at t_start
y = np.array([[P.z0]])
while t < P.t_end:  # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    # updates control and dynamics at faster simulation rate
    while t < t_next_plot:  
        # Get referenced inputs from signal generators
        z_ref = z_refSig.square(t)      
        u, state_hat = controller.update(z_ref, y)
        y = mass.update(u)  # Propagate the dynamics
        t += P.Ts  # advance time by Ts
    # update animation and data plots
    animation.update(mass.state)
    dataPlot.update(t, mass.state, u, z_ref)
    dataPlotObserver.update(t, mass.state, state_hat)
    plt.pause(0.0001)  # allows time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
