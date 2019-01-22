import numpy as np
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
    def __str__(self):
        s = super().__str__()+'\n'
        n_params = 0
        for p in self.parameters():
            n_params += np.prod(p.size())
        s += 'Total params: {}'.format(n_params)
        return s
    def print_summary(self):
        s = str(self)
        print(s)

def one_hot(x, depth, dtype=torch.float32):
    """Convert a batch of indices to a batch of one-hot vectors

    Parameters
    ----------
    depth : int
        The length of each output vector
    """
    i = x.unsqueeze(-1).expand(-1, depth)
    return torch.zeros_like(i, dtype=dtype).scatter_(-1, i, 1)
