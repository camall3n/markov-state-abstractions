import numpy as np
import torch
import torch.nn.functional as F

from ..nn.qnet import QNet
from ..nn.nnutils import one_hot
from .replaymemory import ReplayMemory, Experience

class DQNAgent():
    def __init__(self, n_latent_dims, n_actions, phi, lr=0.001, epsilon=0.05, batch_size=16, train_phi=False):
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.lr = lr
        self.phi = phi
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.copy_period = 50
        self.n_steps_init = 500#batch_size*3
        self.decay_period = 2500
        self.train_phi = train_phi
        self.replay = ReplayMemory(10000)
        self.reset()

    def reset(self):
        self.n_training_steps = 0
        self.n_hidden_layers = 1
        self.q = QNet(n_features=self.n_latent_dims, n_actions=self.n_actions, n_hidden_layers=self.n_hidden_layers)
        self.q_target = QNet(n_features=self.n_latent_dims, n_actions=self.n_actions, n_hidden_layers=self.n_hidden_layers)
        self.copy_target_net()
        self.replay.reset()
        params = list(self.q.parameters()) + list(self.phi.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)


    def get_epsilon(self):
        alpha = (len(self.replay) - self.n_steps_init)/self.decay_period
        alpha = np.clip(alpha, 0, 1)
        return self.epsilon*alpha + 1*(1-alpha)

    def act(self, x):
        if (len(self.replay) < self.n_steps_init
            or np.random.uniform() < self.get_epsilon()):
                a = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                q_values = self.q(self.phi(torch.tensor(x, dtype=torch.float32)))
                a = torch.argmax(q_values, dim=-1).numpy().tolist()[0]
        return a

    def v(self, x):
        with torch.no_grad():
            z = self.phi(torch.stack(tch(x, dtype=torch.float32)))
            v = self.q(z).numpy().max(axis=-1)
        return v

    def train(self, x, a, r, xp, done, gamma):
        self.replay.push(x, a, r, xp, done)

        if len(self.replay) < self.n_steps_init:
            return

        experiences = self.replay.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        with torch.no_grad():
            zp = self.phi(torch.stack(tch(batch.xp, dtype=torch.float32)))
            # Compute Double-Q targets
            ap = torch.argmax(self.q(zp), dim=-1)
            vp = self.q_target(zp).gather(-1, ap.unsqueeze(-1)).squeeze(-1)
            # vp = torch.max(self.q(zp),dim=-1)[0]
            not_done_idx = (1-torch.stack(tch(batch.done, dtype=torch.float32)))
            targets = torch.stack(tch(batch.r, dtype=torch.float32)) + gamma*vp*not_done_idx

        self.q.train()
        self.optimizer.zero_grad()
        z = self.phi(torch.stack(tch(batch.x, dtype=torch.float32)))
        if not self.train_phi:
            z = z.detach()
        q_values = self.q(z)
        q_targets_full = q_values.clone().detach()
        for i, a in enumerate(batch.a):
            q_targets_full[i,a] = targets[i]

        loss = F.smooth_l1_loss(input=q_values, target=q_targets_full)
        loss.backward()
        self.optimizer.step()

        self.n_training_steps += 1
        if self.copy_period is not None and self.n_training_steps % self.copy_period == 0:
            self.copy_target_net()
        # self.soft_update()

        return loss.detach().numpy().tolist()

    def copy_target_net(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def soft_update(self, tau=0.1):
        """
        Soft update of target network from policy network.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (nn.Module): weights will be copied from
            target_model (nn.Module): weights will be copied to
            tau (float): interpolation parameter - usually small eg 0.0001
        """

        for theta_target, theta in zip(self.q_target.parameters(), self.q.parameters()):
            theta_target.data.copy_(tau * theta.data + (1.0 - tau) * theta_target.data)

def tch(tensor, dtype=torch.float32):
    return list(map(lambda x: torch.tensor(x, dtype=dtype), tensor))
