import argparse
import glob
import os
import json

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# sns.set(style="darkgrid")

algs = ['truestate', 'rearrange_xy']

dfs = []
for alg in algs:
    for filepath in glob.glob('scores/train_0k/{}/*/*.txt'.format(alg)):
        data = pd.read_json(filepath, lines=True)
        data['alg'] = 'true-xy' if alg == 'truestate' else 'rearranged-xy'
        seed = int(filepath.split('/')[-2].split('_seed_')[-1])
        data['seed'] = seed
        dfs.append(data)
data = pd.concat(dfs, axis=0)

len(data.query('alg == "true-xy"')['seed'].unique())
len(data.query('alg == "rearranged-xy"')['seed'].unique())

sns.relplot(
    data=data,
    x='episode',
    y='reward',
    hue='alg',
    style='alg',
    style_order=['true-xy', 'rearranged-xy'],
    kind='line',
    # units='seed_number',
    # estimator=None,
)
plt.tight_layout()
plt.savefig('rearranged-xy.png')
plt.show()
