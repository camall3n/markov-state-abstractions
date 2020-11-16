import argparse
import glob
import json
import numpy as np
import os
import random
from sklearn.neighbors import KernelDensity
import torch

def reset_seeds(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def get_parser():
    """Return a nicely formatted argument parser

    This function is a simple wrapper for the argument parser I like to use,
    which has a stupidly long argument that I always forget.
    """
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def load_experiment(tag, coefs=None):
    logfiles = sorted(glob.glob(os.path.join('logs', tag + '*', 'train-*.txt')))
    seeds = [f.split('-')[-1].split('.')[0] for f in logfiles]
    logs = [open(f, 'r').read().splitlines() for f in logfiles]

    def read_log(log, coefs=coefs):
        results = [json.loads(item) for item in log]
        fields = results[0].keys()
        data = dict([(f, np.asarray([item[f] for item in results])) for f in fields])
        if coefs is None:
            coefs = {
                'L_inv': 1.0,
                'L_fwd': 0.1,
                'L_cpc': 1.0,
                'L_fac': 0.1,
            }
        if 'L' not in fields:
            data['L'] = sum([
                coefs[f] * data[f] if f != 'L_fac' else coefs[f] * (data[f] - 1)
                for f in coefs.keys()
            ])
        return data

    results = [read_log(log) for log in logs]
    data = dict(zip(seeds, results))
    return data
