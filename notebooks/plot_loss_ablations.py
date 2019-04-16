import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from gridworlds.utils import load_experiment

def load_experiment(tag):
    logfiles = sorted(glob.glob(os.path.join('logs',tag+'*','train-*.txt')))
    seeds = [int(f.split('-')[-1].split('.')[0]) for f in logfiles]
    logs = [open(f,'r').read().splitlines() for f in logfiles]
    def read_log(log):
        results = [json.loads(item) for item in log]
        data = pd.DataFrame(results)
        return data
    results = [read_log(log) for log in logs]
    data = pd.concat(results, join='outer', keys=seeds, names=['seed']).sort_values(by='seed', kind='mergesort').reset_index(level=0)
    return data

labels = ['tag', 'factored', 'fwd_model']
experiments = [('exp6-no-fac-no-fwd', False, False),
               ('exp7-no-fac',        False, True),
               ('exp8-no-fwd-6x6',     True, False),
               ('exp9-6x6',            True, True)]
data = pd.concat([load_experiment(e[0]) for e in experiments], join='outer', keys=experiments, names=labels).reset_index(level=[0,1,2])

g = sns.relplot(x='step', y='MI', units='seed', row='fwd_model', col='factored', kind='line', estimator=None, data=data, hue='tag', alpha=0.2, legend=False, height=4)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Normalized Mutual Information')
plt.show()

#%%
g = sns.relplot(x='step', y='L', units='seed', row='fwd_model', col='factored', kind='line', estimator=None, data=data, hue='tag', alpha=0.2, legend=False, height=4)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Total Loss vs. Time')
plt.show()

#%%
g = sns.relplot(x='L', y='MI', units='seed', row='fwd_model', col='factored', kind='line', estimator=None, hue='tag', data=data, alpha=0.3, height=4, legend=False)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Mutual Information vs. Loss')
plt.show()

#%%
g = sns.relplot(x='step', y='L_fac', units='seed', row='fwd_model', col='factored', kind='line', estimator=None, data=data, hue='tag', alpha=0.2, legend=False, height=4)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('L_fac vs. Time')
plt.show()

#%%

labels =          ['tag', 'cpc_model']
experiments = [('exp10-no-cpc-6x6', False),
               ('exp9-6x6',         True)]
data = pd.concat([load_experiment(e[0]) for e in experiments], join='outer', keys=experiments, names=labels).reset_index(level=[0,1])

g = sns.relplot(x='step', y='MI', units='seed', col='cpc_model', kind='line', estimator=None, data=data, hue='tag', alpha=0.2, legend=False, height=4)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Normalized Mutual Information')
plt.show()

#%%
g = sns.relplot(x='step', y='L', units='seed', col='cpc_model', kind='line', estimator=None, data=data, hue='tag', alpha=0.2, legend=False, height=4)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Total Loss vs. Time')
plt.show()

#%%
g = sns.relplot(x='L', y='MI', units='seed', col='cpc_model', kind='line', estimator=None, data=data, hue='tag', alpha=0.2, legend=False, height=4)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('MI vs. L')
plt.show()

#%%
g = sns.relplot(x='step', y='L_fac', units='seed', col='cpc_model', kind='line', estimator=None, data=data, hue='tag', alpha=0.2, legend=False, height=4)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('L_fac vs. Time')
plt.show()
