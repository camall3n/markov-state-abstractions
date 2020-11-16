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
