import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class RepVisualization:
    def __init__(self, env, x0, x1, obs, colors=None, cmap=None):
        self.env = env
        self.fig = plt.figure(figsize=(10,8))
        self.cmap = cmap
        self.colors = colors

        self._plot_states(x0, subplot=331, title='states (t)')
        self._plot_states(x1, subplot=333, title='states (t+1)')

        ax = self.fig.add_subplot(332)
        ax.imshow(obs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('observations (t)')

        ax = self.fig.add_subplot(337)
        env.plot(ax)

        z0 = np.zeros_like(x0)
        z1_hat = np.zeros_like(x0)
        z1 = np.zeros_like(x0)

        _, self.inv_sc = self._plot_rep(z0, subplot=334, title=r'$\phi(x_t)$')
        ax, self.fwd_sc = self._plot_rep(z1_hat, subplot=335, title=r'$T(\phi(x_t),a_t)$')
        _, self.true_sc = self._plot_rep(z1, subplot=336, title=r'$\phi(x_{t+1})$')

        self.tstep = ax.text(-0.75, .7, 'updates = '+str(0))
        self.tinv = ax.text(-0.75, .5, 'inv_loss = '+str(np.nan))
        self.tfwd = ax.text(-0.75, .3, 'fwd_loss = '+str(np.nan))
        self.tdis = ax.text(-0.75, .1, 'dis_loss = '+str(np.nan))
        self.tent = ax.text(-0.75,-.1, 'ent_loss = '+str(np.nan))

        self.effects_hat = self._setup_effects(subplot=338)
        self.effects = self._setup_effects(subplot=339)

        self.fig.tight_layout(pad=5.0, h_pad=1.08, w_pad=2.5)
        self.fig.show()

    def _plot_states(self, x, subplot=111, title=''):
        ax = self.fig.add_subplot(subplot)
        ax.scatter(x[:,1],-x[:,0],c=self.colors, cmap=self.cmap)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    def _plot_rep(self, z, subplot=111, title=''):
        ax = self.fig.add_subplot(subplot)
        x = z[:,0]
        y = z[:,1]
        sc = ax.scatter(x,y,c=self.colors, cmap=self.cmap)
        ax.set_xlim([-1.1,1.1])
        ax.set_ylim([-1.1,1.1])
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
        ax.set_ylim([-2,2])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def _plot_effects(self, z0, z1, a, ax, title='', noise=False):
        ax.clear()
        ax.set_xlabel('action')
        ax.set_ylabel(r'$\Delta\ z$')
        ax.set_ylim([-2,2])
        ax.set_title(title)
        n_dims = z0.shape[-1]
        dz_flat = (z1 - z0).flatten()
        if noise:
            dz_flat += noise * np.random.randn(len(dz_flat))
        a_flat = np.repeat(a, n_dims)
        var_flat = np.tile(np.arange(n_dims), len(a))
        sns.violinplot(x=a_flat, y=dz_flat, hue=var_flat, inner=None, dodge=False, bw='silverman', ax=ax)
        ax.axhline(y=0, ls=":", c=".5")

        # Re-label legend entries
        for i, t in enumerate(ax.legend_.texts):
            t.set_text(r'$z_{('+str(i)+')}$')
        plt.setp(ax.collections, alpha=.7)
        return ax

    def update_plots(self, step, z0, z1_hat, z1, inv_loss, fwd_loss, dis_loss, ent_loss, a, a_hat):
        self.inv_sc.set_offsets(z0)
        self.fwd_sc.set_offsets(z1_hat)
        self.true_sc.set_offsets(z1)

        self.tstep.set_text('updates = '+str(step))
        self.tinv.set_text('inv_loss = '+str(inv_loss))
        self.tfwd.set_text('fwd_loss = '+str(fwd_loss))
        self.tdis.set_text('dis_loss = '+str(dis_loss))
        self.tent.set_text('ent_loss = '+str(ent_loss))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self._plot_effects(z0, z1_hat, a, ax=self.effects_hat, title=r'$T(\phi(x_t),a) - \phi(x_{t})$', noise=0.02)
        self._plot_effects(z0, z1, a, ax=self.effects, title=r'$\phi(x_{t+1}) - \phi(x_{t})$', noise=0.02)

        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return frame
