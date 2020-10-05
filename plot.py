import argparse
import glob
import os
import json

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# sns.set(style="darkgrid")


def smooth_and_bin(data, bin_size, window_size):
    numeric_dtypes = data.dtypes.apply(pd.api.types.is_numeric_dtype)
    numeric_cols = numeric_dtypes.index[numeric_dtypes]
    data[numeric_cols] = data[numeric_cols].rolling(window_size).mean()
    # starting from window_size, get every bin_size row
    data = data[window_size::bin_size]
    return data


def parse_filepath(fp, filename, bin_size, window_size):
    try:
        data = pd.read_csv(os.path.join(fp, filename))
        # data = smooth_and_bin(data, bin_size, window_size)
        with open(os.path.join(fp, 'params.json'), "r") as json_file:
            params = json.load(json_file)
        for k, v in params.items():
            data[k] = v
        return data
    except FileNotFoundError as e:
        print("Error in parsing filepath {fp}: {e}".format(fp=fp, e=e))
        return None


def collate_results(results_dirs, filename, bin_size, window_size):
    dfs = []
    for run in results_dirs:
        print("Found {run}".format(run=run))
        run_df = parse_filepath(run, filename, bin_size, window_size)
        if run_df is None:
            continue
        dfs.append(run_df)
    return pd.concat(dfs, axis=0)


def plot(data, x, y, hue, style, col, seed, savepath=None, show=True):
    print("Plotting using hue={hue}, style={style}, {seed}".format(hue=hue, style=style, seed=seed))
    assert not data.empty, "DataFrame is empty, please check query"

    print(data.query('episode==99').groupby('agent', as_index=False)['total_reward'].mean())
    print(data.query('episode==99').groupby('agent', as_index=False)['total_reward'].std())

    # If asking for multiple envs, use facetgrid and adjust height
    height = 4 if col is not None and len(data[col].unique()) > 1 else 5
    if col:
        col_wrap = 2 if len(data[col].unique()) > 1 else 1
    else:
        col_wrap = None

    palette = sns.color_palette('Set1', n_colors=len(data[hue].unique()), desat=0.5)
    # dashes = {
    #         'RBF-DQN'      : '',
    #         'ICNN'         : (1, 1),
    #         'NAF'          : (1, 2, 5, 2),
    #         'RBF-DDPG'     : (2, 2, 1, 2),
    #         'DDPG'         : (5, 2, 5, 2),
    #         'TD3'          : (7, 2, 3, 2),
    #         'SAC'          : (1, 2, 3, 2),
    #         }

    # col_order = ['SymbolicProcgen_4x4', 'SymbolicProcgen_4x8', 'SymbolicProcgen_8x16']

    # palette_value = {
    #         'RBF-DQN'      : 'red',
    #         'ICNN'         : 'green',
    #         'NAF'          : 'orange',
    #         'CAQL'         : 'blue',
    #     }

    # palette_sota = {
    #         'RBF-DDPG'     : 'red',
    #         'RBF-DQN'      : 'red',
    #         'DDPG'         : 'blue',
    #         'TD3'          : 'green',
    #         'SAC'          : 'black',
    #         'CAQL'         : 'magenta',
    #     }

    # palette = palette_sota
    # labels = ['DDPG', 'TD3', 'SAC', 'RBF-DQN']
    # labels = ['DDPG', 'RBF-DDPG']
    labels = [
            'true-state-dqn',
            'pretrained-phi-dqn',
            'end-to-end-dqn',
            'random',
            'no-Linv-dqn',
            'no-Lrat-dqn',
            ]
    if isinstance(seed, list) or seed == 'average':
        g = sns.relplot(x=x,
                        y=y,
                        data=data,
                        hue=hue,
                        hue_order=labels,
                        style=style,
                        kind='line',
                        # legend='full',
                        legend=False,
                        # dashes=dashes,
                        height=height,
                        aspect=1.2,
                        col=col,
                        col_wrap=col_wrap,
                        # col_order=col_order,
                        palette=palette,
                        facet_kws={'sharey': False, 'sharex': False})

    elif seed == 'all':
        g = sns.relplot(x=x,
                        y=y,
                        data=data,
                        hue=hue,
                        units='seed',
                        style=style,
                        estimator=None,
                        kind='line',
                        legend='full',
                        height=height,
                        aspect=1.2,
                        col=col,
                        col_wrap=col_wrap,
                        palette=palette,
                        facet_kws={'sharey': False, 'sharex': False})
    else:
        raise ValueError("{seed} not a recognized choice".format(seed=seed))

    g.set_titles('{col_name}')

    # g.axes.flat[-2].legend(labels, bbox_to_anchor=(0.5, -0.5), loc='lower center', ncol=len(labels))
    # g.axes.flat[-2].legend(labels, loc='lower right')
    g.axes.flat[0].set_ylim((-100,0))
    g.axes.flat[0].legend(labels, loc='best')
    plt.tight_layout()

    # Adding temp results
    # line_res = {
    #     'Ant-v3': {
    #             'DDPG': 888.77,
    #             'TD3': 4372.44,
    #             'SAC': 4000,
    #         },
    #     'HalfCheetah-v3': {
    #             'DDPG': 8577.29,
    #             'TD3': 9636.95,
    #             'SAC': 13000,
    #         },
    #     'Hopper-v3': {
    #             'DDPG': 1400,
    #             'TD3': 3000,
    #             'SAC': 3200,
    #         },
    #     }

    # for env, ax in map(lambda ax: (ax.title.get_text(), ax), g.axes):
    #     if env in line_res.keys():
    #         for agent, value in line_res[env].items():
    #             print(agent, value)
    #             if agent != 'SAC' and env in ['HalfCheetah-v3']:
    #                 ax.axhline(value, xmax=0.5, dashes=dashes[agent], color=palette[agent])
    #             elif agent == 'CAQL' and env in ['Hopper-v3']:
    #                 ax.axhline(value, xmax=0.5, dashes=dashes[agent], color=palette[agent])
    #             else:
    #                 ax.axhline(value, dashes=dashes[agent], color=palette[agent])

    if savepath is not None:
        g.savefig(savepath)

    if show:
        plt.show()


def parse_args():
    # Parse input arguments
    # Use --help to see a pretty description of the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # yapf: disable
    parser.add_argument('--results-dirs', help='Directories for results', required=True, nargs='+', type=str)
    parser.add_argument('--filename', help='CSV filename', required=False, type=str, default='reward.csv')
    parser.add_argument('--bin-size', help='How much to reduce the data by', type=int, default=10)
    parser.add_argument('--window-size', help='How much to average the data by', type=int, default=10)

    parser.add_argument('-x', help='Variable to plot on x axis', required=False, type=str, default='episode')
    parser.add_argument('-y', help='Variable to plot on y axis', required=False, type=str, default='reward')

    parser.add_argument('--query', help='DF query string', type=str)
    parser.add_argument('--hue', help='Hue variable', type=str)
    parser.add_argument('--style', help='Style variable', type=str)
    parser.add_argument('--col', help='Column variable', type=str)
    parser.add_argument('--seed', help='How to handle seeds', type=str, default='average')

    parser.add_argument('--no-plot', help='No plots', action='store_true')
    parser.add_argument('--no-show', help='Does not show plots', action='store_true')
    parser.add_argument('--savepath', help='Save the plot here', type=str)
    # yapf: enable

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Looking for logs in results directory")
    print("Smoothing by {window_size}, binning by {bin_size}".format(window_size=args.window_size,
                                                                     bin_size=args.bin_size))
    assert args.filename is not None, "Must pass filename if creating csv"
    df = collate_results(args.results_dirs, args.filename, args.bin_size, args.window_size)
    df = df.convert_dtypes(convert_string=False, convert_integer=False)
    bool_cols = df.dtypes[df.dtypes == 'boolean'].index
    df = df.replace(to_replace={k: pd.NA for k in bool_cols}, value={k: False for k in bool_cols})
    if not args.no_plot:
        assert args.x is not None and args.y is not None, "Must pass x, y if creating csv"
        if args.savepath:
            os.makedirs(os.path.split(args.savepath)[0], exist_ok=True)
        if args.query is not None:
            print("Filtering with {query}".format(query=args.query))
            df = df.query(args.query)
        plot(df,
             args.x,
             args.y,
             args.hue,
             args.style,
             args.col,
             args.seed,
             savepath=args.savepath,
             show=(not args.no_show))
