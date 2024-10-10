import numpy as np
import matplotlib.pyplot as plt
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter
from hummingbirdDynamics import HummingbirdDynamics
from ctrlEquilibrium import ctrlEquilibrium

hover = 0.5

# instantiate hummingBird, controller, and reference classes
hummingBird = HummingbirdDynamics(alpha=0.0)
controller = ctrlEquilibrium()
x_eqlbm = np.array([[0],
                    [0]])   # theta, thetadot
# F_L = SignalGenerator(amplitude=0.01, frequency=0.1, y_offset=hover-0.02)
# F_r = SignalGenerator(amplitude=0.01, frequency=0.101, y_offset=hover-0.02)

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
        # u = controller.update(x_eqlbm)
        u = controller.update2(x_eqlbm, y)
        y = hummingBird.update(u)  # Propagate the dynamics
        t += P.Ts  # advance time by Ts
    # update animation and data plots at rate t_plot
    animation.update(t, hummingBird.state)
    dataPlot.update(t, hummingBird.state, u)
    plt.pause(0.01)  # allows time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
# plt.waitforbuttonpress()
# plt.close()
