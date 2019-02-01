import matplotlib.pyplot as plt
import numpy as np

class RepVisualization:
    def __init__(self, x0, x1, obs, colors=None, cmap='Set3'):
        self.fig = plt.figure(figsize=(10,6))
        self.cmap = cmap
        self.colors = colors

        self._plot_states(x0, subplot=231, title='states (t)')
        self._plot_states(x1, subplot=233, title='states (t+1)')

        ax = self.fig.add_subplot(232)
        ax.imshow(obs)
        plt.xticks([])
        plt.yticks([])
        ax.set_title('observations (t)')

        z0 = np.zeros_like(x0)
        z1_hat = np.zeros_like(x0)
        z1 = np.zeros_like(x0)

        _, self.inv_sc = self._plot_rep(z0, subplot=234, title=r'$\phi(x_t)$')
        ax, self.fwd_sc = self._plot_rep(z1_hat, subplot=235, title=r'$T(\phi(x_t),a_t)$')
        _, self.true_sc = self._plot_rep(z1, subplot=236, title=r'$\phi(x_{t+1})$')

        self.tstep = ax.text(-0.75, .7, 'updates = '+str(0))
        self.tinv = ax.text(-0.75, .5, 'inv_loss = '+str(np.nan))
        self.tfwd = ax.text(-0.75, .3, 'fwd_loss = '+str(np.nan))

        self.fig.show()

    def _plot_states(self, x, subplot=111, title=''):
        ax = self.fig.add_subplot(subplot)
        ax.scatter(x[:,0],x[:,1],c=self.colors, cmap=self.cmap)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks([])
        plt.yticks([])
        ax.set_title(title)

    def _plot_rep(self, z, subplot=111, title=''):
        ax = self.fig.add_subplot(subplot)
        x = z[:,0]
        y = z[:,1]
        sc = ax.scatter(x,y,c=self.colors, cmap=self.cmap)
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])
        plt.xlabel(r'$z_0$')
        plt.ylabel(r'$z_1$')
        plt.xticks([])
        plt.yticks([])
        ax.set_title(title)
        return ax, sc

    def update_plots(self, step, z0, z1_hat, z1, inv_loss, fwd_loss):
        self.inv_sc.set_offsets(z0)
        self.fwd_sc.set_offsets(z1_hat)
        self.true_sc.set_offsets(z1)

        self.tstep.set_text('updates = '+str(step))
        self.tinv.set_text('inv_loss = '+str(inv_loss))
        self.tfwd.set_text('fwd_loss = '+str(fwd_loss))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return frame
