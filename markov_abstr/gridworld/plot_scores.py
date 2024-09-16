import glob
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

from visgrid.utils import get_parser

parser = get_parser()
# yapf: disable
parser.add_argument('experiment_name', type=str,
                    help='The name of the experiment for which to plot scores')
parser.add_argument('--smoothing', type=int, default=5,
                    help='Number of data points for sliding window average')
# yapf: enable
args = parser.parse_args()


def load_experiment(path):
    logfiles = sorted(glob.glob(os.path.join(path, "scores-*.txt")))
    agents = [path.split("/")[-2] for f in logfiles]
    seeds = [int(f.split("-")[-1].split(".")[0]) for f in logfiles]
    logs = [open(f, "r").read().splitlines() for f in logfiles]

    def read_log(log):
        results = [json.loads(item) for item in log]
        data = smooth(pd.DataFrame(results), args.smoothing)
        return data

    results = [read_log(log) for log in logs]
    keys = list(zip(agents, seeds))
    data = (
        pd.concat(results, join="outer", keys=keys, names=["agent", "seed"])
        .sort_values(by="seed", kind="mergesort")
        .reset_index(level=[0, 1])
    )
    return data


def smooth(data, n):
    numeric_dtypes = data.dtypes.apply(pd.api.types.is_numeric_dtype)
    numeric_cols = numeric_dtypes.index[numeric_dtypes]
    data[numeric_cols] = data[numeric_cols].rolling(n).mean()
    return data


experiments = [f"{args.experiment_name}"]

root = "results/scores/"
unfiltered_paths = [(root + e + "/", (e)) for e in experiments]
experiments = [
    experiment for path, experiment in unfiltered_paths if os.path.exists(path)
]
paths = [path for path, _ in unfiltered_paths if os.path.exists(path)]
labels = ["tag", "features"]
data = pd.concat(
    [load_experiment(p) for p in paths], join="outer", keys=(experiments), names=labels
).reset_index(level=list(range(len(labels))))


def plot(data, x, y, hue, style, col=None):
    print("Plotting using hue={hue}, style={style}".format(hue=hue, style=style))
    assert not data.empty, "DataFrame is empty, please check query"

    data = data.replace("truestate", "xy-position")

    print(data.groupby("agent", as_index=False)["reward"].mean())
    print(data.groupby("agent", as_index=False)["reward"].std())

    # If asking for multiple envs, use facetgrid and adjust height
    height = 4 if col is not None and len(data[col].unique()) > 1 else 5

    labels = [
        "Expert (x,y)",
    ]
    colormap = [
        "xy-position",
    ]
    p = sns.color_palette("Set1", n_colors=2)
    red, _ = p

    p = sns.color_palette("Set1", n_colors=9, desat=0.5)
    _, blue, green, purple, orange, yellow, brown, pink, gray = p

    palette = [red, blue, brown, purple, orange, yellow, pink]
    palette = dict(zip(colormap, palette))

    g = sns.relplot(
        x=x,
        y=y,
        data=data,
        kind="line",
        legend=False,
        height=height,
        aspect=1.2,
        linewidth=2,
        facet_kws={"sharey": False, "sharex": False},
    )

    g.set_titles("{col_name}")

    ax = g.axes.flat[0]
    ax.set_ylim((-90, 0))
    ax.set_xlim((0, 100))
    leg = ax.legend(
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.43, -0.17),
        fontsize=12,
        frameon=False,
    )
    leg.set_draggable(True)
    for axis in ["bottom", "left"]:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(width=2)
    ax.tick_params(labelsize=16)
    ax.set_ylabel("Reward", fontsize=18)
    ax.set_xlabel("Episode", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()


plot(data, x="episode", y="reward", hue="agent", style="agent")
