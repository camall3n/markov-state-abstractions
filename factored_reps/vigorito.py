import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import time

XY_DIMS = 6
UNIFORM = XY_DIMS+0
CONSTANT = UNIFORM+1
SEQUENCE = CONSTANT+1

class VigoritoWorld:
    def __init__(self):
        self.reset()
        self.n_actions = XY_DIMS
        self.n_states = len(self.state)

    def reset(self):
        self.state = np.zeros(9)
        self.state[:XY_DIMS] = np.random.uniform(0, 1, size=6)
        self.state[UNIFORM]  = np.random.uniform(0, 1)
        self.state[CONSTANT] = 0.5
        self.state[SEQUENCE] = 0.5

    def reset_agent(self):
        self.agent.position = self.get_random_position()
        at = lambda x, y: np.all(x.position == y.position)
        while (self.goal is not None) and at(self.agent, self.goal):
            self.agent.position = self.get_random_position()

    def step(self, action):
        assert len(action==6)
        self.state[:XY_DIMS] += action + np.random.normal(0, 0.01, size=6)
        self.state[UNIFORM] = np.random.uniform(0,1)
        b = self.state[SEQUENCE] + np.random.normal(0.05, 0.001)
        self.state[:XY_DIMS] = np.clip(self.state[:XY_DIMS],0,1)
        self.state[SEQUENCE] = (b+1) if (b < 0) else (b-1) if (b > 1) else b

        s = self.get_state()
        r = 0
        done = False
        return s, r, done

    def get_state(self):
        return np.copy(self.state)

    def plot(self, axes=None):
        n_subplots = XY_DIMS//2 + 3
        if axes is not None:
            assert len(axes) == n_subplots
            fig = None
        else:
            fig, axes = plt.subplots(nrows=1, ncols=(XY_DIMS//2+3), figsize=(6,2),
                gridspec_kw={'width_ratios': [1]*(XY_DIMS//2)+[0.33]*3})
        for i in range(XY_DIMS//2):
            axes[i].set_xlim([0,1])
            axes[i].set_ylim([0,1])
            axes[i].scatter(self.state[2*i],self.state[2*i+1])
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_xlabel('XY({})'.format(i))
        line_labels = ['UNIF', 'CONST', 'SEQ']
        for i in range(n_subplots-XY_DIMS//2):
            plot_line(self.state[XY_DIMS+i], axes[XY_DIMS//2+i])
            axes[XY_DIMS//2+i].set_xlabel(line_labels[i])
        return fig, axes

def plot_line(y, ax):
    xmid=0.5
    ymin=0
    ymax=1
    bar_width=1
    ax.set_xlim(0,1)
    ax.set_ylim(ymin,ymax)
    ax.vlines(xmid, ymin, ymax)
    ax.plot(xmid, y, 'ro', mfc='r')
    sns.despine(ax=ax, left=True, right=True, top=False, bottom=False)
    ax.set_yticks([])
    ax.set_xticks([])

def run_agent(env, n_samples=1000, video=False):
    if video:
        fig, axes = env.plot()
        fig.show()
    states = [env.get_state()]
    actions = []
    for _ in range(n_samples):
        # a = np.random.uniform(-0.1,0.1,size=6)# <--- continuous actions
        a = np.random.choice([-0.1,0,0.1], size=6)# <--- discrete actions
        env.step(a)
        actions.append(a)
        states.append(env.get_state())

        if video:
            for ax in axes:
                ax.clear()
            env.plot(axes)
            fig.canvas.draw()
            fig.canvas.flush_events()
    return np.stack(states,axis=0), np.stack(actions,axis=0)

#%%
if __name__ == '__main__':
    env = VigoritoWorld()
    run_agent(env, 1000, video=True)
