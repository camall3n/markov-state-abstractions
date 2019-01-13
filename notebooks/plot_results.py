%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

envname = 'busytaxi5x5'
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
