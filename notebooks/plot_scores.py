import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from gridworlds.utils import load_experiment

def load_experiment(tag):
    logfiles = sorted(glob.glob(os.path.join('scores',tag+'*','scores-*.txt')))
    agents = [f.split('-')[-2] for f in logfiles]
    seeds = [int(f.split('-')[-1].split('.')[0]) for f in logfiles]
    logs = [open(f,'r').read().splitlines() for f in logfiles]
    def read_log(log):
        results = [json.loads(item) for item in log]
        data = pd.DataFrame(results)
        return data
    results = [read_log(log) for log in logs]
    keys = list(zip(agents, seeds))
    data = pd.concat(results, join='outer', keys=keys, names=['agent','seed']).sort_values(by='seed', kind='mergesort').reset_index(level=[0,1])
    return data

labels = ['tag']
experiments = [
    'test-6x6-random',
    'test-6x6-dqn-true-state',
    # 'test-6x6-dqn-phi-train',
    'test-nofac-6x6-dqn-train-phi',
    'test-6x6-dqn-phi-nofac']
data = pd.concat([load_experiment(e) for e in experiments], join='outer', keys=experiments, names=labels).reset_index(level=[0])

g = sns.relplot(x='episode', y='total_reward', kind='line', hue='tag', units='trial', estimator=None, data=data, height=4, alpha=0.2, col='tag', col_wrap=2, legend=False)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Reward vs. Time')
plt.show()
