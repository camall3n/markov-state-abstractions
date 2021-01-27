import matplotlib.pyplot as plt
import seaborn as sns

from rbfdqn.utils_for_q_learning import get_hyper_parameters
from dmcontrol import plotkit

#%% Test loading experiments

#%% Filter and plot the data
sns.set(rc={
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'figure.facecolor': 'white',
})

domains = [
    # 'CartpoleSwingup',
    # 'CheetahRun',
    'FingerSpin',
    # 'PendulumSwingup',
    'WalkerWalk',
]
experiments = [
    'representation_gap',
]

# Plot the data
for domain in domains:
    full_domain = 'dm2gym-' + domain + '-v0'
    run_paths = [
        'dmcontrol/experiments/{}/{}/*'.format(full_domain, experiment)
        for experiment in experiments
    ]

    try:
        data = plotkit.load_globbed_experiments(*run_paths,
                                                results_file='loss.csv',
                                                header=['total_loss', 'rbf_loss', 'markov_loss'])
    except ValueError:
        continue
    data = data.query('ep > 0')
    p = sns.color_palette('viridis', n_colors=len(data['markov_coef'].unique()), desat=0.5)
    p[0] = (0, 0, 0)
    sns.relplot(
        data=data,
        x='ep',
        y='markov_loss',
        hue='markov_coef',
        # style='',
        kind='line',
        # units='seed_number',
        # estimator=None,
        palette=p)
    plt.title(domain)
    # plt.ylim([0, 1000])
    plt.tight_layout()
    # save_plot('representation_gap/', domain)
    plt.show()
