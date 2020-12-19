import os

import torch
import numpy as np

from gym import spaces

from gridworlds.nn import nnutils

def conv2d_size_out(size, kernel_size, stride):
    ''' Adapted from pytorch tutorials:
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''
    return ((size[-2] - (kernel_size[-2] - 1) - 1) // stride + 1,
            (size[-1] - (kernel_size[-1] - 1) - 1) // stride + 1)

def build_phi_network(params, input_shape, mode='rainbow-de'):
    """
    Description:
        Construct the appropriate kind of phi network for the pretraining step in the markov
        abstractions

    Parameters:
        - args : Namespace
            See the `markov` folder for more information on the argparse
        - input_shape : Tuple[Int]
            Shape of the input (state generally)
    """

    if mode == 'curl':
        final_size = conv2d_size_out(input_shape, (3, 3), 2)
        final_size = conv2d_size_out(final_size, (3, 3), 1)
        final_size = conv2d_size_out(final_size, (3, 3), 1)
        final_size = conv2d_size_out(final_size, (3, 3), 1)
        output_size = final_size[0] * final_size[1] * 32
        phi = nnutils.Sequential(
            torch.nn.Conv2d(params['frame_stack'], 32, kernel_size=(4, 4), stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(),
            nnutils.Reshape(-1, output_size),
            torch.nn.Linear(output_size, params['latent_dim']),
            torch.nn.LayerNorm(params['latent_dim']),
            torch.nn.Tanh(),
        )
    elif mode == 'rainbow':
        final_size = conv2d_size_out(input_shape, (8, 8), 4)
        final_size = conv2d_size_out(final_size, (4, 4), 2)
        final_size = conv2d_size_out(final_size, (3, 3), 1)
        output_size = final_size[0] * final_size[1] * 64
        phi = nnutils.Sequential(
            torch.nn.Conv2d(params['frame_stack'], 32, kernel_size=(8, 8), stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(),
            nnutils.Reshape(-1, output_size),
            torch.nn.Linear(output_size, params['latent_dim']),
            torch.nn.LayerNorm(params['latent_dim']),
            torch.nn.Tanh(),
        )
    elif mode == 'rainbow-de':
        final_size = conv2d_size_out(input_shape, (5, 5), 5)
        final_size = conv2d_size_out(final_size, (5, 5), 5)
        output_size = final_size[0] * final_size[1] * 64
        phi = nnutils.Sequential(
            torch.nn.Conv2d(params['frame_stack'], 32, kernel_size=(5, 5), stride=5),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(5, 5), stride=5),
            torch.nn.ReLU(),
            nnutils.Reshape(-1, output_size),
            torch.nn.Linear(output_size, params['latent_dim']),
            torch.nn.LayerNorm(params['latent_dim']),
            torch.nn.Tanh(),
        )

    return phi

class InverseModel(torch.nn.Module):
    """
    Description:
        Network module that captures predicting the action given a state, next_state pair.

    Parameters:
        - params
        - n_actions : Int
            The number of actions in the environment

    """
    def __init__(self, params, n_actions, discrete=False):
        super(InverseModel, self).__init__()
        self.discrete = discrete
        self.body = torch.nn.Sequential(
            torch.nn.Linear(params['latent_dim'] * 2, params['layer_size']),
            torch.nn.ReLU(),
            torch.nn.Linear(params['layer_size'], params['layer_size']),
            torch.nn.ReLU(),
        )
        if self.discrete:
            self.log_pr_linear = torch.nn.Linear(params['layer_size'], n_actions)
        else:
            self.mean_linear = torch.nn.Linear(params['layer_size'], n_actions)
            self.log_std_linear = torch.nn.Linear(params['layer_size'], n_actions)

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        shared_vector = self.body(context)

        if self.discrete:
            return self.log_pr_linear(shared_vector)
        else:
            mean = self.mean_linear(shared_vector)
            log_std = self.log_std_linear(shared_vector)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            std = log_std.exp()
            return mean, std

class ContrastiveModel(torch.nn.Module):
    """
    Description:
        Network module that captures if a given state1, state2 pair belong in the same transition.

    Parameters:
        - params
    """
    def __init__(self, params):
        super(ContrastiveModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(params['latent_dim'] * 2, params['layer_size']),
            torch.nn.ReLU(),
            torch.nn.Linear(params['layer_size'], 1),
        )

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        return self.model(context)

class MarkovHead(torch.nn.Module):
    """
    Description:
        Network module that combines contrastive and inverse models.

    Parameters:
        - params
        - action_space : Int
            The environment's action space
    """
    def __init__(self, params, action_space):
        super(MarkovHead, self).__init__()
        self.discrete = (action_space is spaces.Discrete)
        if self.discrete:
            self.n_actions = action_space.n
        else:
            assert len(action_space.shape) == 1
            self.n_actions = action_space.shape[0]

        self.inverse_model = InverseModel(params, self.n_actions, discrete=self.discrete)
        self.discriminator = ContrastiveModel(params)

        self.bce = torch.nn.BCEWithLogitsLoss()
        if self.discrete:
            self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_markov_loss(self, z0, z1, a):
        # Inverse loss
        if self.discrete:
            log_pr_actions = self.inverse_model(z0, z1)
            l_inverse = self.ce(input=log_pr_actions, target=a)
        else:
            mean, std = self.inverse_model(z0, z1)
            cov = torch.diag_embed(std, dim1=1, dim2=2)
            normal = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
            log_pr_action = normal.log_prob(a)
            l_inverse = -1 * log_pr_action.mean(dim=0)

        # Ratio loss
        with torch.no_grad():
            N = len(z1)
            idx = torch.randperm(N)  # shuffle indices of next states
        z1_neg = z1.view(N, -1)[idx].view(z1.size())

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        is_real_transition = torch.cat([torch.ones(N), torch.zeros(N)], dim=0).to(z0.device)
        log_pr_real = self.discriminator(z0_extended, z1_pos_neg)
        l_ratio = self.bce(input=log_pr_real, target=is_real_transition.unsqueeze(-1).float())

        markov_loss = l_inverse + l_ratio
        return markov_loss

class FeatureNet(nnutils.Network):
    def __init__(self, params, action_space, input_shape):
        super(FeatureNet, self).__init__()
        self.phi = build_phi_network(params, input_shape, mode=params['encoder_type'])
        self.markov_head = MarkovHead(params, action_space)

        if params['optimizer'] == 'RMSprop':
            make_optimizer = torch.optim.RMSprop
        elif params['optimizer'] == 'Adam':
            make_optimizer = torch.optim.Adam
        else:
            raise NotImplementedError('unknown optimizer')
        self.optimizer = make_optimizer(self.parameters(), lr=params['learning_rate'])

    def forward(self, x):
        return self.phi(x)

    def loss(self, batch):
        states, actions, _, next_states, _ = batch
        markov_loss = self.markov_head.compute_markov_loss(
            z0=self.phi(torch.as_tensor(states.astype(np.float32))),
            z1=self.phi(torch.as_tensor(next_states.astype(np.float32))),
            a=torch.as_tensor(actions, dtype=torch.int64),
        )
        loss = markov_loss
        return loss

    def save_phi(self, path, name):
        full_path = os.path.join(path, name)
        torch.save((self.phi, self.feature_size), full_path)
        return full_path

    def train_one_batch(self, batch):
        loss = self.loss(batch)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()
