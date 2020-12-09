import numpy
import os
import torch


def action_checker(env):
	for l, h in zip(env.action_space.low, env.action_space.high):
		if l != -h:
			print("asymetric action space")
			print("don't know how to deal with it")
			assert False
	if numpy.max(env.action_space.low) != numpy.min(env.action_space.low):
		print("different action range per dimension")
		assert False
	if numpy.max(env.action_space.high) != numpy.min(env.action_space.high):
		print("different action range per dimension")
		assert False


def get_hyper_parameters(name, alg):
	meta_params = {}
	with open(alg + "_hyper_parameters/" + name + ".hyper") as f:
		lines = [line.rstrip('\n') for line in f]
		for l in lines:
			parameter_name, parameter_value, parameter_type = (l.split(','))
			if parameter_type == 'string':
				meta_params[parameter_name] = str(parameter_value)
			elif parameter_type == 'integer':
				meta_params[parameter_name] = int(parameter_value)
			elif parameter_type == 'float':
				meta_params[parameter_name] = float(parameter_value)
			else:
				print("unknown parameter type ... aborting")
				print(l)
				sys.exit(1)
	return meta_params


def sync_networks(target, online, alpha, copy=False):
	if copy == True:
		for online_param, target_param in zip(online.parameters(), target.parameters()):
			target_param.data.copy_(online_param.data)
	elif copy == False:
		for online_param, target_param in zip(online.parameters(), target.parameters()):
			target_param.data.copy_(alpha * online_param.data +
			                        (1 - alpha) * target_param.data)


def save(li_returns, li_loss, params, alg):
	directory = alg + "_results/" + params['hyper_parameters_name'] + '/'
	if not os.path.exists(directory):
		os.makedirs(directory)
	numpy.savetxt(directory + str(params['seed_number']) + ".txt", li_returns)

	directory = alg + "_results/" + params['hyper_parameters_name'] + '/loss_'
	numpy.savetxt(directory + str(params['seed_number']) + ".txt", li_loss)


def set_random_seed(meta_params):
	seed_number = meta_params['seed_number']
	import numpy
	numpy.random.seed(seed_number)
	import random
	random.seed(seed_number)
	import torch
	torch.manual_seed(seed_number)
	meta_params['env'].seed(seed_number)
	meta_params['env'].action_space.np_random.seed(seed_number)


class Reshape(torch.nn.Module):
	"""
	Description:
		Module that returns a view of the input which has a different size    Parameters:
		- args : Int...
			The desired size
	"""
	def __init__(self, *args):
		super().__init__()
		self.shape = args

	def __repr__(self):
		s = self.__class__.__name__
		s += '{}'.format(self.shape)
		return s

	def forward(self, x):
		return x.view(*self.shape)
