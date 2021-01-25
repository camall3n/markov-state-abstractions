import argparse
import glob
import os
import json

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(rc={
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'figure.facecolor': 'white',
})


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
        if bin_size != 0 and window_size != 0:
            data = smooth_and_bin(data, bin_size, window_size)
        # with open(os.path.join(fp, 'params.json'), "r") as json_file:
        #     params = json.load(json_file)
        params = pd.read_csv(os.path.join(fp, 'hyperparams.csv'),
                index_col='param',
                names=['param', 'value', 'type'])
        params = params[['value']].to_dict()['value']
        for k, v in params.items():
            data[k] = v
        return data
    except (FileNotFoundError, NotADirectoryError) as e:
        # print("Error in parsing filepath {fp}: {e}".format(fp=fp, e=e))
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
    # If asking for multiple envs, use facetgrid and adjust height
    height = 3 if col is not None and len(data[col].unique()) > 2 else 5
    if col:
        col_wrap = 2 if len(data[col].unique()) > 1 else 1
    else:
        col_wrap = None
    col_order = ['small', 'medium', 'large', 'giant'] if col == 'model_shape' else None

    palette = sns.color_palette('Set1', n_colors=len(data[hue].unique()), desat=0.5)
    if isinstance(seed, list) or seed == 'average':
        g = sns.relplot(x=x,
                        y=y,
                        data=data,
                        hue=hue,
                        style=style,
                        kind='line',
                        legend='full',
                        height=height,
                        aspect=1.5,
                        col=col,
                        col_wrap=col_wrap,
                        col_order=col_order,
                        palette=palette,
                        facet_kws={'sharey': False})

    elif seed == 'all':
        g = sns.relplot(x=x,
                        y=y,
                        data=data,
                        hue=hue,
                        units='seed_number',
                        style=style,
                        estimator=None,
                        kind='line',
                        legend='full',
                        height=height,
                        aspect=1.5,
                        col=col,
                        col_wrap=col_wrap,
                        col_order=col_order,
                        palette=palette,
                        facet_kws={'sharey': False})
    else:
        raise ValueError("{seed} not a recognized choice".format(seed=seed))

    if savepath is not None:
        g.savefig(savepath)

    g.set(**{'ylim': (0, 1000)})

    if show:
        plt.show()


def parse_args():
    # Parse input arguments
    # Use --help to see a pretty description of the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # yapf: disable
    parser.add_argument('--results-dirs', help='Directories for results', required=True, nargs='+', type=str)
    parser.add_argument('--filename', help='CSV filename', required=False, type=str)
    parser.add_argument('--bin-size', help='How much to reduce the data by', type=int, default=0)
    parser.add_argument('--window-size', help='How much to average the data by', type=int, default=0)

    parser.add_argument('-x', help='Variable to plot on x axis', required=False, type=str)
    parser.add_argument('-y', help='Variable to plot on y axis', required=False, type=str)

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

    if not args.no_plot:
        assert args.x is not None and args.y is not None, "Must pass x, y if creating csv"
        if args.savepath:
            os.makedirs(os.path.split(args.savepath)[0], exist_ok=True)
#        args.query = "(env_name == 'dm2gym:CartpoleSwingup-v0' and temperature == '1.0' and learning_rate == '0.0001' and features == 'expert') or (env_name == 'dm2gym:CheetahRun-v0' and temperature == '0.1' and learning_rate == '0.0003' and features == 'expert') or (env_name == 'dm2gym:FingerSpin-v0' and temperature == '1.0' and learning_rate == '0.0003' and features == 'expert') or (env_name == 'dm2gym:PendulumSwingup-v0' and temperature == '1.0' and learning_rate == '0.001' and features == 'expert') or (env_name == 'dm2gym:WalkerWalk-v0' and temperature == '1.0' and learning_rate == '0.0001' and features == 'expert') or (env_name == 'dm2gym:CartpoleSwingup-v0' and temperature == '2.0' and learning_rate == '0.0001' and features == 'visual') or (env_name == 'dm2gym:CheetahRun-v0' and temperature == '0.5' and learning_rate == '0.0001' and features == 'visual') or (env_name == 'dm2gym:FingerSpin-v0' and temperature == '0.5' and learning_rate == '0.0001' and features == 'visual') or (env_name == 'dm2gym:PendulumSwingup-v0' and temperature == '0.5' and learning_rate == '0.001' and features == 'visual') or (env_name == 'dm2gym:WalkerWalk-v0' and temperature == '0.5' and learning_rate == '0.0001' and features == 'visual')"
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
