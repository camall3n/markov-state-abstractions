import json
from glob import glob
import os
import pandas as pd

for experiment in ['train_0k', 'train_30k']:
    root = 'results/scores/{}/'.format(experiment)
    for agent in [
            'markov',
            'inv-only',
            'contr-only',
            'autoenc',
            'truestate',
            'end-to-end',
            'pixel-pred',
    ]:
        pattern = '/scores-*.txt' if agent == 'pixel-pred' else '/*/scores-*.txt'
        for fname in glob(root + agent + pattern):
            print(fname)
            if agent == 'pixel-pred':
                d = os.path.splitext(fname)[0]
                os.makedirs(d, exist_ok=True)
            else:
                d = os.path.dirname(fname)
            b = os.path.splitext(os.path.basename(fname))[0]
            with open(fname, 'r') as f:
                df = []
                for l in f:
                    df.append(json.loads(l))
                df = pd.DataFrame(df)
            df.to_csv(d + '/reward.csv')
            p = {
                'agent': agent,
                'seed': b.split('-')[-1],
                'env': 'GridWorld-6x6',
            }
            with open(d + '/params.json', 'w') as f:
                json.dump(p, f)
