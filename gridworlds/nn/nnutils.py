import logging
import shutil

import numpy as np
import os
import torch
import torch.nn

class Reshape(torch.nn.Module):
    """Module that returns a view of the input which has a different size

    Parameters
    ----------
    args : int...
        The desired size
    """
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def __repr__(self):
        s = self.__class__.__name__
        s += '{}'.format(self.shape)
        return s

    def forward(self, input):
        return input.view(*self.shape)

class Network(torch.nn.Module):
    """Module that, when printed, shows its total number of parameters
    """
    def __init__(self):
        super().__init__()
        self.frozen = False

    def __str__(self):
        s = super().__str__() + '\n'
        n_params = 0
        for p in self.parameters():
            n_params += np.prod(p.size())
        s += 'Total params: {}'.format(n_params)
        return s

    def print_summary(self):
        s = str(self)
        print(s)

    def save(self, name, model_dir, is_best=False):
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, '{}_latest.pytorch'.format(name))
        torch.save(self.state_dict(), model_file)
        logging.info('Model saved to {}'.format(model_file))
        if is_best:
            best_file = os.path.join(model_dir, '{}_best.pytorch'.format(name))
            shutil.copyfile(model_file, best_file)
            logging.info('New best model! Model copied to {}'.format(best_file))

    def load(self, model_file, force_cpu=False):
        logging.info('Loading model from {}...'.format(model_file))
        map_loc = 'cpu' if force_cpu else None
        state_dict = torch.load(model_file, map_location=map_loc)
        self.load_state_dict(state_dict)

    def freeze(self):
        if not self.frozen:
            for param in self.parameters():
                param.requires_grad = False
            self.frozen = True

    def unfreeze(self):
        if self.frozen:
            for param in self.parameters():
                param.requires_grad = True
            self.frozen = False

class Sequential(torch.nn.Sequential, Network):
    pass

def one_hot(x, depth, dtype=torch.float32):
    """Convert a batch of indices to a batch of one-hot vectors

    Parameters
    ----------
    depth : int
        The length of each output vector
    """
    i = x.unsqueeze(-1).expand(-1, depth)
    return torch.zeros_like(i, dtype=dtype).scatter_(-1, i, 1)

def extract(input, idx, idx_dim, batch_dim=0):
    '''
Extracts slices of input tensor along idx_dim at positions
specified by idx.

Notes:
    idx must have the same size as input.shape[batch_dim].
    Output tensor has the shape of input with idx_dim removed.

Args:
    input (Tensor): the source tensor
    idx (LongTensor): the indices of slices to extract
    idx_dim (int): the dimension along which to extract slices
    batch_dim (int): the dimension to treat as the batch dimension

Example::

    >>> t = torch.arange(24, dtype=torch.float32).view(3,4,2)
    >>> i = torch.tensor([1, 3, 0], dtype=torch.int64)
    >>> extract(t, i, idx_dim=1, batch_dim=0)
        tensor([[ 2.,  3.],
                [14., 15.],
                [16., 17.]])
'''
    if idx_dim == batch_dim:
        raise RuntimeError('idx_dim cannot be the same as batch_dim')
    if len(idx) != input.shape[batch_dim]:
        raise RuntimeError(
            "idx length '{}' not compatible with batch_dim '{}' for input shape '{}'".format(
                len(idx), batch_dim, list(input.shape)))
    viewshape = [
        1,
    ] * input.ndimension()
    viewshape[batch_dim] = input.shape[batch_dim]
    idx = idx.view(*viewshape).expand_as(input)
    result = torch.gather(input, idx_dim, idx).mean(dim=idx_dim)
    return result
