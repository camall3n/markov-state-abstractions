# ---------------------------------------------------------------
# This hack prevents a matplotlib/dm_control crash on macOS.
# We open/close a plot before importing dm_control.suite.
import matplotlib.pyplot as plt
plt.plot()
plt.close()
from dm_control import suite
# ---------------------------------------------------------------

env = suite.load('cartpole', 'swingup')
pixels = env.physics.render()
plt.imshow(pixels)
plt.show()
