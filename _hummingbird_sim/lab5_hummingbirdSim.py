import numpy as np
import matplotlib.pyplot as plt
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter
from hummingbirdDynamics import HummingbirdDynamics
from ctrlLonPD import ctrlLonPD

hover = 0.5

# instantiate hummingBird, controller, and reference classes
hummingBird = HummingbirdDynamics(alpha=0.0)
controller = ctrlLonPD()
theta_refSig = SignalGenerator(amplitude=0.1, frequency=0.1)

# instantiate the simulation plots and animation
dataPlot = DataPlotter()
animation = HummingbirdAnimation()

u = np.array([[0], [0]], dtype=np.float64)
y = np.array([[P.phi0], [P.theta0], [P.psi0]])
t = P.t_start  # time starts at t_start
while t < P.t_end:  # main simulation loop
    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        theta_ref = theta_refSig.square(t)
        u = controller.update(theta_ref, y)
        y = hummingBird.update(u)  # Propagate the dynamics
        t += P.Ts  # advance time by Ts
    # update animation and data plots at rate t_plot
    animation.update(t, hummingBird.state)
    dataPlot.update(t, hummingBird.state, u, np.array([[0],[theta_ref],[0]]))
    plt.pause(0.01)  # allows time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
# plt.waitforbuttonpress()
# plt.close()
