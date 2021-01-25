from argparse import Namespace
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
        params = pd.read_csv(os.path.join(fp, 'hyperparams.csv'),
                index_col='param',
                names=['param', 'value', 'type'])
        params = params[['value']].to_dict()['value']
        for k, v in params.items():
            data[k] = v
        return data
    except (FileNotFoundError, NotADirectoryError) as e:
        return None


def collate_results(results_dir, filename, bin_size, window_size):
    dfs = []
    for run in glob.glob(results_dir):
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


def setup():
    # Default stuff
    args = Namespace(
            results_dir='dmcontrol/experiments/*/*/**/*',
            filename='scores.csv',
            bin_size=0,
            window_size=0,
            no_show=False,
        )

    # Stuff to edit per plot
    args.query = ' '.join([
        "(env_name == 'dm2gym:CartpoleSwingup-v0' and temperature == '1.0' and learning_rate == '0.0001' and features == 'expert')",
        "or (env_name == 'dm2gym:CheetahRun-v0' and temperature == '0.1' and learning_rate == '0.0003' and features == 'expert')",
        "or (env_name == 'dm2gym:FingerSpin-v0' and temperature == '1.0' and learning_rate == '0.0003' and features == 'expert')",
        "or (env_name == 'dm2gym:PendulumSwingup-v0' and temperature == '1.0' and learning_rate == '0.001' and features == 'expert')",
        "or (env_name == 'dm2gym:WalkerWalk-v0' and temperature == '1.0' and learning_rate == '0.0001' and features == 'expert')",
        "or (env_name == 'dm2gym:CartpoleSwingup-v0' and temperature == '2.0' and learning_rate == '0.0001' and features == 'visual')",
        "or (env_name == 'dm2gym:CheetahRun-v0' and temperature == '0.5' and learning_rate == '0.0001' and features == 'visual')",
        "or (env_name == 'dm2gym:FingerSpin-v0' and temperature == '0.5' and learning_rate == '0.0001' and features == 'visual')",
        "or (env_name == 'dm2gym:PendulumSwingup-v0' and temperature == '0.5' and learning_rate == '0.001' and features == 'visual')",
        "or (env_name == 'dm2gym:WalkerWalk-v0' and temperature == '0.5' and learning_rate == '0.0001' and features == 'visual')",
        ])
    args.savepath = None # './images/' -- change if you want to save


    # Plotting args
    args.x = 'episode'
    args.y = 'reward'
    args.col = 'env_name'
    args.hue = 'features'
    args.style = 'features'
    args.seed = 'average'

    return args


if __name__ == '__main__':
    args = setup()

    # Now plotting
    print("Looking for logs in results directory")

    df = collate_results(args.results_dir, args.filename, args.bin_size, args.window_size)

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
