import numpy
from cpprb import ReplayBuffer, create_env_dict, create_before_add_func
import random


class buffer_class:
	def __init__(self, max_length, seed_number, env):
		env_dict = create_env_dict(env)
		self.before_add = create_before_add_func(env)
		self.storage = ReplayBuffer(max_length, env_dict)

	def append(self, s, a, r, done, sp):
		self.storage.add(**self.before_add(obs=s, act=a, rew=r, done=done, next_obs=sp))

	def sample(self, batch_size):
		batch = self.storage.sample(batch_size)
		s_matrix = batch['obs']
		a_matrix = batch['act']
		r_matrix = batch['rew']
		done_matrix = batch['done']
		sp_matrix = batch['next_obs']
		return s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix

	def __len__(self):
		return self.storage.get_stored_size()
