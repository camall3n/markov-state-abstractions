import glob
import logging
import os
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rbfdqn.utils_for_q_learning import get_hyper_parameters

logging.basicConfig(level=logging.INFO)

#%%

def load_runs(path, results_file='scores.csv', params_file='hyperparams.csv'):
    dfs = []
    for trial_path in glob.glob(os.path.join(path, "seed_*")):
        _, subdirs, subfiles = next(os.walk(trial_path))
        if results_file not in subfiles:
            logging.warning("Results file '{}' missing from {}".format(results_file, trial_path))
            continue
        if params_file not in subfiles:
            logging.warning("Params file '{}' missing from {}".format(params_file, trial_path))
            continue

        data = pd.read_csv(os.path.join(trial_path, results_file))
        data.index.name = 'episode (x10)'
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

experiment_path = 'dmcontrol/experiments/dm2gym-CartpoleSwingup-v0/representation_gap/'
# d = load_experiment(experiment_path, tag_list=['expert_3447aa9_5__learningrate_0.00075'])
# d = load_experiment(experiment_path, tag_list=['*_5__*', '*_6__*'])
# d = load_experiment(experiment_path)
d = load_experiment(experiment_path, tag_list=['tuning-*'])
d.learning_rate.unique()
d.temperature.unique()
sorted(d.columns)

def save_plot(experiment, domain):
    plot_path = os.path.join('dmcontrol/plots', experiment)
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, domain.lower() + '.png'))

#%%
domains = [
    'CartpoleSwingup',
    'CheetahRun',
    'FingerSpin',
    'PendulumSwingup',
    'WalkerWalk',
]

basepath = 'dmcontrol/experiments'
experiment = 'representation_gap'
sns.set(rc={
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'figure.facecolor': 'white',
})
for domain, ax in zip(domains, axes):
    domain_path = os.path.join(basepath, 'dm2gym-' + domain + '-v0')
    experiment_path = os.path.join(domain_path, experiment)
    try:
        data = load_experiment(experiment_path, tag_list=['tuning-*'])
    except ValueError:
        continue
    data = data.query('learning_rate==0.0001')
    p = sns.color_palette('Set1', n_colors=len(data['temperature'].unique()), desat=0.5)
    sns.relplot(data=data,
                x='episode',
                y='reward',
                hue='temperature',
                style='temperature',
                kind='line',
                palette=p)
    plt.title(domain)
    plt.ylim([0, 1000])
    plt.tight_layout()
    save_plot('tuning-lr/tuning-lr-0.0001', domain)
    plt.show()
