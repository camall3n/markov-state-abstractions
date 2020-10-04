import json
from glob import glob
import os
import pandas as pd

size = '6x6'

for fname in glob('scores/'+size+'/pretrained/*/*.txt'):
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
        'agent': 'pretrained-phi-dqn',
        'seed': b.split('-')[-1],
        'env': 'GridWorld-'+size,
        }
    with open(d + '/params.json', 'w') as f:
        json.dump(p, f)

for fname in glob('scores/'+size+'/end-to-end/*/*.txt'):
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
        'agent': 'end-to-end-dqn',
        'seed': b.split('-')[-1],
        'env': 'GridWorld-'+size,
        }
    with open(d + '/params.json', 'w') as f:
        json.dump(p, f)

for fname in glob('scores/'+size+'/random/*/*.txt'):
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
        'agent': 'random',
        'seed': b.split('-')[-1],
        'env': 'GridWorld-'+size,
        }
    with open(d + '/params.json', 'w') as f:
        json.dump(p, f)

for fname in glob('scores/'+size+'/truestate/*/*.txt'):
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
        'agent': 'true-state-dqn',
        'seed': b.split('-')[-1],
        'env': 'GridWorld-'+size,
        }
    with open(d + '/params.json', 'w') as f:
        json.dump(p, f)
