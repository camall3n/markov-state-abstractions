%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from gridworlds.domain.taxi.taxi import Taxi5x5, BusyTaxi5x5

env = Taxi5x5()
envname = env.name
env.plot()
plt.show()
#%%
skilledq = np.loadtxt('results/'+envname+'/skilledqlearning/results.txt')
random = np.loadtxt('results/'+envname+'/random/results.txt')
t = np.arange(len(skilledq))
plt.plot(t, skilledq, label='q-learning w/ skills')
plt.plot(t, random, label='random w/ skills')
plt.title(envname)
plt.xlabel('timesteps')
plt.ylabel('total reward')
plt.legend()
plt.show()
