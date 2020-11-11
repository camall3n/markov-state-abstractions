import json
from glob import glob
import os
import pandas as pd

root = 'scores/camera_3k/train_dqn_'
size = '6x6'
time = '_3000'

for fname in glob(root + size + time + '_*/*.txt'):
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
        'agent': 'markov',
        'seed': b.split('-')[-1],
        'env': 'GridWorld-' + size,
    }
    with open(d + '/params.json', 'w') as f:
        json.dump(p, f)

for fname in glob(root + size + time + 'auto*/*.txt'):
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
        'agent': 'autoencoder',
        'seed': b.split('-')[-1],
        'env': 'GridWorld-' + size,
    }
    with open(d + '/params.json', 'w') as f:
        json.dump(p, f)

for fname in glob(root + size + time + 'end_to_end*/*.txt'):
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
        'agent': 'visual',
        'seed': b.split('-')[-1],
        'env': 'GridWorld-' + size,
    }
    with open(d + '/params.json', 'w') as f:
        json.dump(p, f)

for fname in glob(root + size + '_' + time + 'random*/*.txt'):
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
        'env': 'GridWorld-' + size,
    }
    with open(d + '/params.json', 'w') as f:
        json.dump(p, f)

for fname in glob(root + size + time + 'truestate*/*.txt'):
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
        'agent': 'xy-position',
        'seed': b.split('-')[-1],
        'env': 'GridWorld-' + size,
    }
    with open(d + '/params.json', 'w') as f:
        json.dump(p, f)

for fname in glob(root + size + time + 'noLinv*/*.txt'):
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
        'agent': 'contrastive',
        'seed': b.split('-')[-1],
        'env': 'GridWorld-' + size,
    }
    with open(d + '/params.json', 'w') as f:
        json.dump(p, f)

for fname in glob(root + size + time + 'noLrat*/*.txt'):
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
        'agent': 'inverse',
        'seed': b.split('-')[-1],
        'env': 'GridWorld-' + size,
    }
    with open(d + '/params.json', 'w') as f:
        json.dump(p, f)
