import json
from glob import glob
import os
import pandas as pd

root = 'scores/loop30k_train_'
size = ''
time = ''

for agent in ['markov', 'inv-only', 'contr-only', 'autoenc']:  #, 'truestate', 'end-to-end']:
    for fname in glob(root + agent + '_*/*.txt'):
        print(fname)
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
