import logging
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import utils_for_q_learning
from gridworlds.nn import nnutils
from dmcontrol.markov import FeatureNet, build_phi_network
from gridworlds.agents.replaymemory import ReplayMemory

from dmcontrol import rad

def rbf_function_on_action(centroid_locations, action, beta):
    '''
    centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
    action_set: Tensor [batch x a_dim (action_size)]
    beta: float
        - Parameter for RBF function

    Description: Computes the RBF function given centroid_locations and one action
    '''
    assert len(centroid_locations.shape) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
    assert len(action.shape) == 2, "Must pass tensor with shape: [batch x a_dim]"

    diff_norm = centroid_locations - action.unsqueeze(dim=1).expand_as(centroid_locations)
    diff_norm = diff_norm**2
    diff_norm = torch.sum(diff_norm, dim=2)
    diff_norm = torch.sqrt(diff_norm + 1e-7)
    diff_norm = diff_norm * beta * -1
    weights = F.softmax(diff_norm, dim=1)  # batch x N
    return weights

def rbf_function(centroid_locations, action_set, beta):
    '''
    centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
    action_set: Tensor [batch x num_act x a_dim (action_size)]
        - Note: pass in num_act = 1 if you want a single action evaluated
    beta: float
        - Parameter for RBF function

    Description: Computes the RBF function given centroid_locations and some actions
    '''
    assert len(centroid_locations.shape) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
    assert len(action_set.shape) == 3, "Must pass tensor with shape: [batch x num_act x a_dim]"

    diff_norm = torch.cdist(centroid_locations, action_set, p=2)  # batch x N x num_act
    diff_norm = diff_norm * beta * -1
    weights = F.softmax(diff_norm, dim=2)  # batch x N x num_act
    return weights

class RBQFNet(nnutils.Network):
    def __init__(self, params, action_space, state_size):
        super().__init__()

        utils_for_q_learning.action_checker(action_space)
        self.params = params
        self.action_space = action_space
        self.action_size = len(action_space.low)
        self.state_size = state_size

        self.N = self.params['num_points']
        self.max_a = self.action_space.high[0]
        self.beta = self.params['temperature']

        self.value_module = nn.Sequential(
            nn.Linear(self.state_size, self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.N),
        )

        if self.params['num_layers_action_side'] == 1:
            self.location_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size_action_side'], self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                nn.Tanh(),
            )
        elif self.params['num_layers_action_side'] == 2:
            self.location_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size_action_side'],
                          self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size_action_side'], self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                nn.Tanh(),
            )

        torch.nn.init.xavier_uniform_(self.location_module[0].weight)
        torch.nn.init.zeros_(self.location_module[0].bias)

        self.location_module[3].weight.data.uniform_(-.1, .1)
        self.location_module[3].bias.data.uniform_(-1., 1.)

        self.criterion = nn.MSELoss()

        self.params_dic = [{
            'params': self.value_module.parameters(),
            'lr': self.params['learning_rate']
        }, {
            'params': self.location_module.parameters(),
            'lr': self.params['learning_rate_location_side']
        }]
        try:
            if self.params['optimizer'] == 'RMSprop':
                self.optimizer = optim.RMSprop(self.params_dic)
            elif self.params['optimizer'] == 'Adam':
                self.optimizer = optim.Adam(self.params_dic)
            else:
                logging.warning('unknown optimizer ....')
        except:
            logging.warning("no optimizer specified ... ")

    def get_centroid_values(self, s):
        '''
        given a batch of s, get all centroid values, [batch x N]
        '''
        centroid_values = self.value_module(s)
        return centroid_values

    def get_centroid_locations(self, s):
        '''
        given a batch of s, get all centroid_locations, [batch x N x a_dim]
        '''
        centroid_locations = self.max_a * self.location_module(s)
        return centroid_locations

    def get_best_qvalue_and_action(self, s):
        '''
        given a batch of states s, return Q(s,a), max_{a} ([batch x 1], [batch x a_dim])
        '''
        all_centroids = self.get_centroid_locations(s)
        values = self.get_centroid_values(s)
        weights = rbf_function(all_centroids, all_centroids, self.beta)  # [batch x N x N]
        allq = torch.bmm(weights, values.unsqueeze(2)).squeeze(2)  # bs x num_centroids
        # a -> all_centroids[idx] such that idx is max(dim=1) in allq
        # a = torch.gather(all_centroids, dim=1, index=indices)
        # (dim: bs x 1, dim: bs x action_dim)
        best, indices = allq.max(dim=1)
        if s.shape[0] == 1:
            index_star = indices.item()
            a = all_centroids[0, index_star]
            return best, a
        else:
            return best, None

    def forward(self, s, a):
        '''
        given a batch of s,a , compute Q(s,a) [batch x 1]
        '''
        centroid_values = self.get_centroid_values(s)  # [batch_dim x N]
        centroid_locations = self.get_centroid_locations(s)
        # [batch x N]
        centroid_weights = rbf_function_on_action(centroid_locations, a, self.beta)
        output = torch.mul(centroid_weights, centroid_values)  # [batch x N]
        output = output.sum(1, keepdim=True)  # [batch x 1]
        return output

    def policy(self, s, epsilon, policy_noise=0):
        '''
        Given state s, at episode, take random action with p=eps
        Note - epsilon is determined by episode and whether training/testing
        '''
        if epsilon > 0 and random.random() < epsilon:
            a = self.action_space.sample()
            return a.tolist()
        else:
            self.eval()
            with torch.no_grad():
                _, a = self.get_best_qvalue_and_action(s)
                a = a.cpu().numpy()
            self.train()
            if policy_noise > 0:
                noise = np.random.normal(loc=0.0, scale=policy_noise, size=len(a))
                a = a + noise
            return a

    def compute_loss(self, Q_target, s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix):
        Q_star, _ = Q_target.get_best_qvalue_and_action(sp_matrix)
        Q_star = Q_star.reshape((self.params['batch_size'], -1))
        with torch.no_grad():
            y = r_matrix + self.params['gamma'] * (1 - done_matrix) * Q_star
        y_hat = self.forward(s_matrix, a_matrix)
        loss = self.criterion(y_hat, y)
        return loss

class Agent:
    def __init__(self, params, env, device):
        self.params = params
        self.device = device
        self.replay_buffer = ReplayMemory(params['max_buffer_size'])

        s0 = env.reset()
        if self.params['data_aug'] == 'crop':
            s0 = rad.center_crop_one_image(s0)
        self.state_shape = s0.shape
        self.feature_type = self.params['features']
        if self.feature_type == 'expert':
            self.encoder = None
        elif self.feature_type == 'visual':
            self.encoder = build_phi_network(params, self.state_shape,
                                             mode=params['encoder_type']).to(device)
        elif self.feature_type == 'markov':
            self.encoder = FeatureNet(params, env.action_space, self.state_shape).to(device)
        else:
            raise NotImplementedError('Unknown feature type')
        if self.encoder is not None:
            print('Encoder:')
            print(self.encoder)
            print()

        s0_matrix = np.array(s0).reshape((1, ) + self.state_shape)
        z0 = self.encode(torch.as_tensor(s0_matrix).float().to(device))
        self.z_dim = len(z0.squeeze(dim=0))

        self.Q_object = RBQFNet(params, env.action_space, self.z_dim).to(device)
        self.Q_object_target = RBQFNet(params, env.action_space, self.z_dim).to(device)
        self.Q_object_target.eval()
        print('Q-Network:')
        print(self.Q_object)
        print()

        utils_for_q_learning.sync_networks(target=self.Q_object_target,
                                           online=self.Q_object,
                                           alpha=params['target_network_learning_rate'],
                                           copy=True)

        policy_type = params['policy_type']
        if policy_type not in ['e_greedy', 'e_greedy_gaussian', 'gaussian']:
            raise NotImplementedError(
                'No get_action function configured for policy type {}'.format(policy_type))
        self.epsilon_schedule = lambda episode: 1.0 / np.power(
            episode, 1.0 / self.params['policy_parameter'])
        self.policy_noise = 0
        # override policy defaults for specific cases
        if policy_type == 'gaussian':
            self.epsilon_schedule = lambda episode: 0
        if policy_type in ['e_greedy_gaussian', 'gaussian']:
            self.policy_noise = self.params['noise']

    def encode(self, state):
        if self.encoder is None:
            return state
        return self.encoder(state)

    def act(self, s, episode, train_or_test):
        if s.shape != self.state_shape:
            s = rad.center_crop_one_image(s, self.state_shape[-1])

        if train_or_test == 'train':
            epsilon = self.epsilon_schedule(episode)
            policy_noise = self.policy_noise
        else:
            epsilon = 0
            policy_noise = 0

        s_matrix = np.array(s).reshape((1, ) + self.state_shape)
        s = torch.from_numpy(s_matrix).float().to(self.device)
        z = self.encode(s)
        return self.Q_object.policy(z, epsilon, policy_noise)

    def store_experience(self, state, action, reward, done, next_state):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self, return_each_loss_info=False):
        if len(self.replay_buffer) < self.params['batch_size']:
            if return_each_loss_info:
                return 0.0, 0.0, 0.0
            else:
                return 0.0
        s_matrix, a_matrix, r_matrix, sp_matrix, done_matrix = self.replay_buffer.sample(
            self.params['batch_size'], itemize=True)
        r_matrix = np.clip(r_matrix,
                           a_min=-self.params['reward_clip'],
                           a_max=self.params['reward_clip'])

        if self.params['data_aug'] is not None:
            if self.params['data_aug'] == 'crop':
                s_matrix = rad.random_crop_image_batch(s_matrix)
                sp_matrix = rad.random_crop_image_batch(sp_matrix)
            else:
                raise NotImplementedError('Unknown data augmentation {}'.format(
                    params['data_aug']))

        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)

        z_matrix = self.encode(s_matrix)
        zp_matrix = self.encode(sp_matrix)

        rbf_loss = self.Q_object.compute_loss(self.Q_object_target, z_matrix, a_matrix, r_matrix,
                                              done_matrix, zp_matrix)
        if self.feature_type == 'markov':
            batch = (z_matrix, a_matrix, r_matrix, zp_matrix, done_matrix)
            markov_loss = self.encoder.loss(batch)
            logging.info('markov_loss = {}'.format(markov_loss.detach().item()))
        else:
            markov_loss = torch.tensor(0.0)
        loss = rbf_loss + self.params['markov_coef'] * markov_loss

        self.Q_object.zero_grad()
        loss.backward()
        self.Q_object.optimizer.step()
        utils_for_q_learning.sync_networks(target=self.Q_object_target,
                                           online=self.Q_object,
                                           alpha=self.params['target_network_learning_rate'],
                                           copy=False)
        if return_each_loss_info:
            return (
                loss.cpu().data.numpy(),
                rbf_loss.cpu().data.numpy(),
                markov_loss.cpu().data.numpy(),
            )
        else:
            return loss.cpu().data.numpy()

    def save(self, is_best=False):
        self.Q_object.save(name='Q_object', model_dir=self.params['models_dir'], is_best=is_best)
        if self.encoder is not None and not self.encoder.frozen:
            self.encoder.save(name='encoder', model_dir=self.params['models_dir'], is_best=is_best)
