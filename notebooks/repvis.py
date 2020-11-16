import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class CleanVisualization:
    def __init__(self, env, obs, batch_size, n_dims, colors=None, cmap=None):
        self.env = env
        self.fig = plt.figure(figsize=(8, 8))
        self.cmap = cmap
        self.colors = colors

        z0 = np.zeros((batch_size, n_dims))
        z1_hat = np.zeros((batch_size, n_dims))
        z1 = np.zeros((batch_size, n_dims))

        plt.rcParams.update({'font.size': 22})

        self.inv_ax, self.inv_sc = self._plot_rep(z0, subplot=111, title=r'$\phi(x)$')

        self.fig.tight_layout()  #pad=5.0, h_pad=1.1, w_pad=2.5)
        self.fig.show()

    def _plot_rep(self, z, subplot=111, title=''):
        ax = self.fig.add_subplot(subplot)
        x = z[:, 0]
        y = z[:, 1]
        sc = ax.scatter(x, y, c=self.colors, cmap=self.cmap)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlabel(r'$z_0$')
        ax.set_ylabel(r'$z_1$')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        return ax, sc

    def update_plots(self, z0, z1_hat, z1, a, a_hat, text):
        self.inv_sc.set_offsets(z0)
        plt.rcParams.update({'font.size': 22})
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3, ))
        return frame
