import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter

# instantiate reference input classes
phi_ref = SignalGenerator(amplitude=1.5, frequency=0.2)
theta_ref = SignalGenerator(amplitude=0.5, frequency=0.2)
psi_ref = SignalGenerator(amplitude=0.5, frequency=0.2)

# instantiate the simulation plots and animation
dataPlot = DataPlotter()
animation = HummingbirdAnimation()

t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # set variables
    phi = phi_ref.sin(t)
    theta = 0#theta_ref.sin(t*5)
    psi = 0#psi_ref.sin(t*5)
    # update animation
    state = np.array([[phi], [theta], [psi], [0.0], [0.0], [0.0]])
    ref = np.array([[0], [0], [0]])
    force = 10
    torque = 10
    pwms = np.array([[force], [torque]])
    animation.update(t, state)
    dataPlot.update(t, state, pwms, ref)

    t = t + P.t_plot  # advance time by t_plot
    plt.pause(0.05)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
