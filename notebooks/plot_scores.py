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

# labels = ['tag','name','factored']
# experiments = [
    # ('test-6x6-random', 'random', True),
    # ('test-6x6-random', 'random', False),
    # 'test-6x6-dqn-true-state',
    # ('test-6x6-dqn-phi-train', 'DQN (end-to-end)'),
    # ('test-6x6-dqn-phi-factored', '2-D factored'),
    # ('test-6x6-dqn-phi-nofac', '2-D unfactored'),
    # 'exp11-3d-rep',
    # ('exp12-10d-rep', '10-D pre-trained'),
    # 'exp13joint10d',
    # ('exp14_true_state', '2-D true state'),
    # 'exp15_pre_2d',
    # ('exp16-onehot36-joint', '36-D one-hot true state'),
    # ('exp17-factored-phi', 'DQN (pre-trained abstraction)'),
    # ('exp18-unfactored-phi', 'DQN (unfactored)'),
# ]
labels = ['tag','name','factored','lr']
experiments = [
    # ('dqn_fqn/dqn1', 'dqn', True),
    # ('dqn_fqn/fqn1', 'fqn', True),
    # ('dqn_fqn/bigdqn', 'big dqn', True),
    ('dqn_fqn/lr-dqn-0.0003', 'dqn', False, 0.0003),
    ('dqn_fqn/lr-dqn-0.003', 'dqn', False, 0.003),
    ('dqn_fqn/lr-dqn-0.001', 'dqn', False, 0.001),
    ('dqn_fqn/lr-dqn-0.01', 'dqn', False, 0.01),
    ('dqn_fqn/lr-dqn-0.03', 'dqn', False, 0.03),
    ('dqn_fqn/lr-dqn-0.1', 'dqn', False, 0.1),
    ('dqn_fqn/lr-dqn-0.0003', 'fqn', True, 0.0003),
    ('dqn_fqn/lr-dqn-0.003', 'fqn', True, 0.003),
    ('dqn_fqn/lr-dqn-0.001', 'fqn', True, 0.001),
    ('dqn_fqn/lr-fqn-0.01', 'fqn', True, 0.01),
    ('dqn_fqn/lr-fqn-0.03', 'fqn', True, 0.03),
    ('dqn_fqn/lr-fqn-0.1', 'fqn', True, 0.1),
]
data = pd.concat([load_experiment(e[0]) for e in experiments], join='outer', keys=experiments, names=labels).reset_index(level=list(range(len(labels))))

# plt.rcParams.update({'font.size': 10})
g = sns.relplot(x='episode', y='total_reward', kind='line', data=data, height=4, alpha=0.2,
    hue='name',
    # style='lr',
    # units='trial', estimator=None,
    col='lr', col_wrap=3,
    # legend=False
)
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Reward vs. Time')
# plt.rcParams.update({'font.size': 22})
plt.savefig('results/foo.png')
plt.show()
