import matplotlib.pyplot as plt
import seaborn as sns

from rbfdqn.utils_for_q_learning import get_hyper_parameters
from dmcontrol import plotkit

#%% Test loading experiments
experiment_path = 'dmcontrol/experiments/dm2gym-CartpoleSwingup-v0/representation_gap/'
experiment_path = 'dmcontrol/experiments/dm2gym-FingerSpin-v0/representation_gap/'
# d = load_experiment(experiment_path, tag_list=['expert_3447aa9_5__learningrate_0.00075'])
# d = load_experiment(experiment_path, tag_list=['*_5__*', '*_6__*'])
# d = load_experiment(experiment_path)
# d = load_experiment(experiment_path, tag_list=['tuning-*'])
d = plotkit.load_experiment(experiment_path, tag_list=['tuning-*'])
d = d.query(
    "(temperature == '0.5' and learning_rate == '0.0001' and features == 'markov' and markov_coef in ['100.0'])"
)
d.learning_rate.unique()
d.temperature.unique()
d.seed_number.unique()
sorted(d.columns)

#%% Filter and plot the data
sns.set(rc={
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'figure.facecolor': 'white',
})

domains = [
    # 'CartpoleSwingup',
    # 'CheetahRun',
    # 'FingerSpin',
    # 'PendulumSwingup',
    'WalkerWalk',
]
experiments = [
    'representation_gap',
    # 'tuning-expert-*',
    # 'tuning-visual-*',
    # 'tuning-markov-*',
    'curl-*',
]

take_all = [
    {'features': 'curl'},
    {'features': 'markov'}
]
best_of = {
    # env_name: [(features, temperature, learning_rate), ...]
    'CartpoleSwingup': [('expert', 1.0, 0.0001), ('visual', 2.0, 0.0001)],
    'CheetahRun': [('expert', 0.1, 0.0003), ('visual', 0.5, 0.0001)],
    'FingerSpin': [('expert', 1.0, 0.0003), ('visual', 0.5, 0.0001)],
    'PendulumSwingup': [('expert', 1.0, 0.001), ('visual', 0.5, 0.001)],
    'WalkerWalk': [('expert', 1.0, 0.0001), ('visual', 0.5, 0.0001)],
}
for domain in best_of.keys():
    best_of[domain] = [{
            'features': features,
            'temperature': temperature,
            'learning_rate': learning_rate,
        } for (features, temperature, learning_rate) in best_of[domain]
    ] + take_all

def build_query(filters):
    query = ' or '.join(['(' +
        ' and '.join([
            field + ' == "{}"'.format(value)
            for (field, value) in filter_.items()
        ])
        + ')' for filter_ in filters
    ])
    return query


# Plot the data
for domain in domains:
    full_domain = 'dm2gym-' + domain + '-v0'
    run_paths = ['dmcontrol/experiments/{}/{}/*'.format(full_domain, experiment)
                    for experiment in experiments]

    try:
        data = plotkit.load_globbed_experiments(*run_paths)
        data.seed_number.unique()
    except ValueError:
        continue
    filters = best_of[domain]
    q = build_query(filters)

    data = data.query(q)
    data.groupby(['learning_rate','features']).size().reset_index()
    # set defaults
    data.loc[((data['features'] == 'curl')
              | (data['features'] == 'expert')
              | (data['features'] == 'visual')), 'markov_coef'] = 0
    p = sns.color_palette('viridis', n_colors=len(data['markov_coef'].unique()), desat=0.5)
    p[0] = (0, 0, 0)
    sns.relplot(
        data=data,
        x='episode',
        y='reward',
        hue='markov_coef',
        style='features',
        kind='line',
        # units='seed_number',
        # estimator=None,
        palette=p)
    plt.title(domain)
    plt.ylim([0, 1000])
    plt.tight_layout()
    # save_plot('representation_gap/', domain)
    plt.show()
