import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from ..visgrid.utils import load_experiment

def load_experiment(tag):
    logfiles = sorted(glob.glob(os.path.join('results/scores', tag + '*', 'scores-*.txt')))
    agents = [f.split('-')[-2] for f in logfiles]
    seeds = [int(f.split('-')[-1].split('.')[0]) for f in logfiles]
    logs = [open(f, 'r').read().splitlines() for f in logfiles]

    def read_log(log):
        results = [json.loads(item) for item in log]
        data = smooth(pd.DataFrame(results), 25)[::25]
        return data

    results = [read_log(log) for log in logs]
    keys = list(zip(agents, seeds))
    data = pd.concat(results, join='outer', keys=keys,
                     names=['agent',
                            'seed']).sort_values(by='seed',
                                                 kind='mergesort').reset_index(level=[0, 1])
    return data  #[data['episode']<=100]

def smooth(data, n):
    numeric_dtypes = data.dtypes.apply(pd.api.types.is_numeric_dtype)
    numeric_cols = numeric_dtypes.index[numeric_dtypes]
    data[numeric_cols] = data[numeric_cols].rolling(n).mean()
    return data

labels = ['tag', 'features', 'grid_type']
experiments = [
    # ('dqn-spiral-markov', 'markov', 'spiral'),
    # ('spiral-relu-no-inv', 'markov-inv+relu', 'spiral'),
    # ('dqn-spiral-expert', 'expert', 'spiral'),
    # ('dqn-spiral-smooth', 'markov+smooth', 'spiral'),
    # ('dqn-spiral-onehot', 'onehot', 'spiral'),
    ('dqn-maze-markov-10k-retrain', 'markov', 'maze'),
    ('maze-relu-no-inv-retrain', 'markov_-inv_+relu', 'maze'),
    ('dqn-maze-expert', 'true-xy', 'maze'),
    # ('dqn-maze-xy-noise', 'noisy-xy', 'maze'),
    # ('dqn-maze-smooth-retrain', 'markov_+relu', 'maze'),
    ('dqn-maze-visual', 'visual', 'maze'),
    # ('dqn-maze-smooth', 'markov+smooth', 'maze'),
    # ('dqn-maze-onehot', 'onehot', 'maze'),
]
experiments = [
    # ('dqn-spiral-markov', 'markov', 'spiral'),
    # ('spiral-relu-no-inv', 'markov-inv+relu', 'spiral'),
    # ('dqn-spiral-expert', 'expert', 'spiral'),
    # ('dqn-spiral-smooth', 'markov+smooth', 'spiral'),
    # ('dqn-spiral-onehot', 'onehot', 'spiral'),
    ('dqn-fixed-maze-coinv1.0-10k', 'markov_coinv+relu', 'maze'),
    ('dqn-retrain-maze-coinv1.0-10k', 'markov_coinv+relu_retrain', 'maze'),
    ('maze-relu-no-inv-retrain', 'markov_-inv_+relu', 'maze'),
    ('dqn-maze-expert', 'true-xy', 'maze'),
    # ('dqn-maze-xy-noise', 'noisy-xy', 'maze'),
    # ('dqn-maze-smooth-retrain', 'markov_+relu', 'maze'),
    ('dqn-maze-visual', 'visual', 'maze'),
    # ('dqn-maze-smooth', 'markov+smooth', 'maze'),
    # ('dqn-maze-onehot', 'onehot', 'maze'),
]
data = pd.concat([load_experiment(e[0]) for e in experiments],
                 join='outer',
                 keys=experiments,
                 names=labels).reset_index(level=list(range(len(labels))))

# plt.rcParams.update({'font.size': 10})
p = sns.color_palette(n_colors=len(data['features'].unique()))
p = sns.color_palette('Set1', n_colors=9, desat=0.5)
red, blue, green, purple, orange, yellow, brown, pink, gray = p
p = [pink, red, orange, yellow, purple]
g = sns.relplot(
    x='episode',
    y='reward',
    kind='line',
    data=data,
    height=4,
    alpha=0.2,
    hue='features',
    style='features',
    # units='seed', estimator=None,
    col='grid_type',  #col_wrap=2,
    # legend=False,
    facet_kws={
        'sharey': True,
        'sharex': False
    },
    palette=p,
)
# plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.15)
g.fig.suptitle('Reward vs. Time')
# plt.rcParams.update({'font.size': 22})
plt.savefig('results/foo.png')
plt.show()
