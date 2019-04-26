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
        s = super().__str__()+'\n'
        n_params = 0
        for p in self.parameters():
            n_params += np.prod(p.size())
        s += 'Total params: {}'.format(n_params)
        return s

    def print_summary(self):
        s = str(self)
        print(s)

    def save(self, tag, name):
        model_dir = 'models/{}'.format(tag)
        os.makedirs(model_dir, exist_ok=True)
        model_file = model_dir+'/{}.pytorch'.format(name)
        torch.save(self.state_dict(), model_file)
        print('Model saved to {}'.format(model_file))

    def load(self, model_file, force_cpu=False):
        print('Loading model from {}...'.format(model_file))
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

def one_hot(x, depth, dtype=torch.float32):
    """Convert a batch of indices to a batch of one-hot vectors

    Parameters
    ----------
    depth : int
        The length of each output vector
    """
    i = x.unsqueeze(-1).expand(-1, depth)
    return torch.zeros_like(i, dtype=dtype).scatter_(-1, i, 1)
