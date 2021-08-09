import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class RepVisualization:
    def __init__(self, env, obs, batch_size, n_dims, colors=None, cmap=None):
        self.env = env
        self.fig = plt.figure(figsize=(10, 8))
        self.cmap = cmap
        self.colors = colors

        self.env_ax = self.fig.add_subplot(331)
        env.plot(self.env_ax)
        self.env_ax.set_title('Environment')

        self.text_ax = self.fig.add_subplot(332)
        self.text_ax.set_xticks([])
        self.text_ax.set_yticks([])
        self.text_ax.axis('off')
        self.text_ax.set_ylim([0, 1])
        self.text_ax.set_xlim([0, 1])
        self.text = self.text_ax.text(0.05, 0.1, '')

        self.obs_ax = self.fig.add_subplot(333)
        self.obs_ax.imshow(obs)
        self.obs_ax.set_xticks([])
        self.obs_ax.set_yticks([])
        self.obs_ax.set_title('Sampled observation (x)')

        z0 = np.zeros((batch_size, n_dims))
        z1_hat = np.zeros((batch_size, n_dims))
        z1 = np.zeros((batch_size, n_dims))

        _, self.inv_sc = self._plot_rep(z0, subplot=334, title=r'$\phi(x_t)$')
        _, self.fwd_sc = self._plot_rep(z1_hat, subplot=335, title=r'$T(\phi(x_t),a_t)$')
        _, self.true_sc = self._plot_rep(z1, subplot=336, title=r'$\phi(x_{t+1})$')

        self.effects_hat = self._setup_effects(subplot=338)
        self.effects = self._setup_effects(subplot=339)

        self.fig.tight_layout(pad=5.0, h_pad=1.1, w_pad=2.5)
        self.fig.show()

    def _plot_states(self, x, subplot=111, title=''):
        ax = self.fig.add_subplot(subplot)
        ax.scatter(x[:, 1], -x[:, 0], c=self.colors, cmap=self.cmap)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

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

    def _setup_effects(self, subplot=111, title=''):
        ax = self.fig.add_subplot(subplot)
        # ax.set_xlabel('action')
        ax.set_ylabel(r'$\Delta\ z$')
        ax.set_ylim([-2, 2])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def _plot_effects(self, z0, z1, a, ax, title='', noise=False):
        ax.clear()
        ax.set_xlabel('action')
        ax.set_ylabel(r'$\Delta\ z$')
        ax.set_ylim([-2, 2])
        ax.set_title(title)
        n_dims = z0.shape[-1]
        dz_flat = (z1 - z0).flatten()
        if noise:
            dz_flat += noise * np.random.randn(len(dz_flat))
        a_flat = np.repeat(a, n_dims)
        var_flat = np.tile(np.arange(n_dims), len(a))
        sns.violinplot(x=a_flat,
                       y=dz_flat,
                       hue=var_flat,
                       inner=None,
                       dodge=False,
                       bw='silverman',
                       ax=ax)
        ax.axhline(y=0, ls=":", c=".5")

        # Re-label legend entries
        for i, t in enumerate(ax.legend_.texts):
            t.set_text(r'$z_{(' + str(i) + ')}$')
        plt.setp(ax.collections, alpha=.7)
        return ax

    def update_plots(self, z0, z1_hat, z1, a, a_hat, text):
        self.inv_sc.set_offsets(z0)
        self.fwd_sc.set_offsets(z1_hat)
        self.true_sc.set_offsets(z1)

        self.text.set_text(text)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self._plot_effects(z0,
                           z1_hat,
                           a,
                           ax=self.effects_hat,
                           title=r'$T(\phi(x_t),a) - \phi(x_{t})$')
        self._plot_effects(z0, z1, a, ax=self.effects, title=r'$\phi(x_{t+1}) - \phi(x_{t})$')

        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3, ))
        return frame

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
