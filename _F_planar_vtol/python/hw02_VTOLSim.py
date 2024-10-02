import matplotlib.pyplot as plt
import numpy as np
import VTOLParam as P
from signalGenerator import signalGenerator
from VTOLAnimation import VTOLAnimation
from dataPlotter import dataPlotter

# instantiate reference input classes, these are not actual values, 
# just values to allow us to plot 
reference = signalGenerator(amplitude=0.5, frequency=0.1)
zSig = signalGenerator(amplitude=0.1, frequency=0.1, y_offset=5)
hSig = signalGenerator(amplitude=1, frequency=0.1, y_offset=5)
thetaSig = signalGenerator(amplitude=0.5, frequency=0.1)
fSig = signalGenerator(amplitude=1, frequency=0.5)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = VTOLAnimation()

t = P.t_start  # time starts at t_start
f = np.array([[0], [0]])
while t < P.t_end:  # main simulation loop
    # set variables
    r = reference.square(t)
    z = zSig.sin(t)
    h = hSig.sin(t)
    theta = thetaSig.sin(t)
    f[0][0] = fSig.sawtooth(t)
    f[1][0] = fSig.sawtooth(t)
    # update animation
    state = np.array([[z], [h], [theta]])  
    animation.update(state)
    dataPlot.update(t, state, f, r)

    # advance time by t_plot
    t += P.t_plot  
    plt.pause(0.01)  # allow time for animation to draw

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
