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
    for filepath in glob.glob('results/scores/train_0k/{}/*/*.txt'.format(alg)):
        data = pd.read_json(filepath, lines=True)
        data['alg'] = 'true-xy' if alg == 'truestate' else 'rearranged-xy'
        seed = int(filepath.split('/')[-2].split('_seed_')[-1])
        data['seed'] = seed
        data.reward = data.reward.rolling(5).mean()
        dfs.append(data)
data = pd.concat(dfs, axis=0)

len(data.query('alg == "true-xy"')['seed'].unique())
len(data.query('alg == "rearranged-xy"')['seed'].unique())

data = data.rename(columns={'alg': 'Features', 'reward': 'Reward', 'episode': 'Episode'})
data.loc[data.Features == 'true-xy', 'Features'] = 'Original (x,y)'
data.loc[data.Features == 'rearranged-xy', 'Features'] = 'Rearranged (x,y)'

p = sns.color_palette('Set1', n_colors=7, desat=0.5)
p[0] = p[5]
p[1] = p[6]
p = p[:2]

g = sns.relplot(
    data=data,
    x='Episode',
    y='Reward',
    hue='Features',
    hue_order=['Original (x,y)', 'Rearranged (x,y)'],
    style='Features',
    style_order=['Rearranged (x,y)', 'Original (x,y)'],
    kind='line',
    # units='seed_number',
    # estimator=None,
    palette=p,
)
leg = g._legend
leg.set_draggable(True)
plt.tight_layout()
plt.savefig('rearranged-xy.png')
plt.show()
