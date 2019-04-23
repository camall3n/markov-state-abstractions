import numpy as np
import matplotlib.pyplot as plt

from gridworlds.domain.gridworld.gridworld import GridWorld
from gridworlds.sensors import *

env = GridWorld(rows=7,cols=4)
env.reset_agent()

sensor = SensorChain([
    OffsetSensor(offset=(0.5,0.5)),
    NoisySensor(sigma=0.1),
    ImageSensor(range=((0,env._rows), (0,env._cols)), pixel_density=3),
    BlurSensor(sigma=0.6, truncate=1.),
])
s = []
for _ in range(100):
    env.step(np.random.randint(4))
    s.append(env.get_state())
s = np.stack(s)
#%%
env.plot()
s = env.get_state()
obs = sensor.observe(s)

plt.figure()
plt.imshow(obs)
plt.xticks([])
plt.yticks([])
plt.show()
