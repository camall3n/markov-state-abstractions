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
    return data[data['episode']<=100]

labels = ['tag','name']
experiments = [
    ('test-6x6-random', 'random'),
    # 'test-6x6-dqn-true-state',
    ('test-6x6-dqn-phi-train', 'DQN (end-to-end)'),
    # ('test-6x6-dqn-phi-factored', '2-D factored'),
    # ('test-6x6-dqn-phi-nofac', '2-D unfactored'),
    # 'exp11-3d-rep',
    # ('exp12-10d-rep', '10-D pre-trained'),
    # 'exp13joint10d',
    # ('exp14_true_state', '2-D true state'),
    # 'exp15_pre_2d',
    # ('exp16-onehot36-joint', '36-D one-hot true state'),
    ('exp17-factored-phi', 'DQN (pre-trained abstraction)'),
    # ('exp18-unfactored-phi', 'DQN (unfactored)'),
]
data = pd.concat([load_experiment(e[0]) for e in experiments], join='outer', keys=experiments, names=labels).reset_index(level=[0,1])

# plt.rcParams.update({'font.size': 22})
g = sns.relplot(x='episode', y='reward', kind='line', hue='name', data=data, height=6, alpha=0.2,
    # units='trial', estimator=None,
    # col='tag', col_wrap=2,
    # legend=False
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Mean episode reward vs. Time (6x6 Gridworld, 150 trials each)')
# plt.rcParams.update({'font.size': 22})
plt.savefig('results/foo.png')
plt.show()
