import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from gridworlds.utils import load_experiment

def load_experiment(tag, coefs=None):
    logfiles = sorted(glob.glob(os.path.join('logs',tag+'*','train-*.txt')))
    seeds = [int(f.split('-')[-1].split('.')[0]) for f in logfiles]
    logs = [open(f,'r').read().splitlines() for f in logfiles]
    def read_log(log, coefs=coefs):
        results = [json.loads(item) for item in log]
        data = pd.DataFrame(results)
        return data
    results = [read_log(log) for log in logs]
    data = pd.concat(results, join='outer', keys=seeds, names=['seed']).sort_values(by='seed', kind='mergesort').reset_index(level=0)
    return data

labels = ['tag', 'fwd_model', 'factored']
experiments = [('exp1-', True, True),
               ('exp4-', True, False),
               ('exp2-', False, True),
               ('exp5-', False, False)]
data = pd.concat([load_experiment(e[0]) for e in experiments], join='outer', keys=experiments, names=labels).reset_index(level=[0,1,2])

g = sns.relplot(x='step', y='MI', units='seed', row='fwd_model', col='factored', kind='line', estimator=None, data=data, hue='tag', alpha=0.2, legend=False, height=4)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Normalized Mutual Information')
plt.show()

#%%
g = sns.relplot(x='step', y='L_fac', units='seed', row='fwd_model', col='factored', kind='line', estimator=None, data=data, hue='tag', alpha=0.2, legend=False, height=4)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('L_fac vs. Time')
plt.show()

#%%
g = sns.relplot(x='step', y='L_inv', units='seed', row='fwd_model', col='factored', kind='line', estimator=None, data=data, hue='tag', alpha=0.2, legend=False, height=4)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('L_inv vs. Time')
plt.show()
