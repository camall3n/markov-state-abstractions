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
    'tuning-visual-features',
]



# Plot the data
for domain in domains:
    full_domain = 'dm2gym-' + domain + '-v0'
    run_paths = [
        'dmcontrol/experiments/{}/{}/*'.format(full_domain, experiment)
        for experiment in experiments
    ]
    # for path in glob.glob(run_paths[0]):pass
    try:
        data = plotkit.load_globbed_experiments(*run_paths,
                                                results_file='loss.csv',
                                                header=['total_loss', 'rbf_loss', 'markov_loss'])
    except ValueError:
        continue
    subset = data.query('temperature == 0.5 and learning_rate == 0.0001')

    # subset = data.query('markov_coef == 1e-5')
    p = sns.color_palette('viridis', n_colors=len(subset['seed_number'].unique()), desat=0.5)
    sns.relplot(
        data=subset,
        x='ep',
        y='total_loss',
        hue='seed_number',
        # style='markov_coef',
        kind='line',
        units='seed_number',
        estimator=None,
        palette=p)
    plt.title(domain)
    # plt.ylim([0, 1000])
    plt.tight_layout()
    # save_plot('representation_gap/', domain)
    plt.yscale('log')
    plt.show()
