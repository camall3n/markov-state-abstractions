%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

skilledq = np.loadtxt('results/skilledqlearning/results.txt')
random = np.loadtxt('results/random/results.txt')
t = np.arange(len(skilledq))
plt.plot(t, skilledq, label='q-learning w/ skills')
plt.plot(t, random, label='random w/ skills')
plt.title('Taxi 5x5')
plt.xlabel('timesteps')
plt.ylabel('total reward')
plt.legend()
plt.show()
