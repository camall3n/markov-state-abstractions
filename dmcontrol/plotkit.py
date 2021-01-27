import glob
import logging
import os
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rbfdqn.utils_for_q_learning import get_hyper_parameters

logging.basicConfig(level=logging.INFO)

def load_runs(path, results_file='scores.csv', params_file='hyperparams.csv', header=None):
    dfs = []
    for trial_path in glob.glob(os.path.join(path, "seed_*")):
        _, subdirs, subfiles = next(os.walk(trial_path))
        if results_file not in subfiles:
            logging.warning("Results file '{}' missing from {}".format(results_file, trial_path))
            continue
        if params_file not in subfiles:
            logging.warning("Params file '{}' missing from {}".format(params_file, trial_path))
            continue

        kwargs = {} if header is None else {'names': header}
        data = pd.read_csv(os.path.join(trial_path, results_file), **kwargs)
        data.index.name = 'ep'
        params = get_hyper_parameters(os.path.join(trial_path, params_file), None)
        for k, v in params.items():
            data[k] = v
        dfs.append(data)
    return pd.concat(dfs, axis=0)

def load_experiment(path, tag_list=None):
    dfs = []
    if tag_list is None:
        tag_list = next(os.walk(path))[1]
    for tag in tag_list:
        tag_path = os.path.join(path, tag)
        data = load_runs(tag_path)
        if data is not None:
            dfs.append(data)
    return pd.concat(dfs, axis=0)

def load_globbed_experiments(*glob_strs, results_file='scores.csv', header=None):
    dfs = []
    for glob_str in glob_strs:
        for path in glob.glob(glob_str):
            data = load_runs(path, results_file=results_file, header=header)
            if data is not None:
                dfs.append(data)
    return pd.concat(dfs, axis=0)

def save_plot(experiment, domain):
    plot_path = os.path.join('dmcontrol/plots', experiment)
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, domain.lower() + '.png'))
