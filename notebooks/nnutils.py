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

class FeatureNet(Network):
    def __init__(self, n_actions, input_shape=2, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32, lr=0.001):
        super().__init__()
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.lr = lr

        self.shape_flat = np.prod(self.input_shape)

        self.phi_layers = []
        self.phi_layers.extend([Reshape(-1, self.shape_flat)])
        self.phi_layers.extend([torch.nn.Linear(self.shape_flat, n_units_per_layer), torch.nn.Tanh()])
        self.phi_layers.extend([torch.nn.Linear(n_units_per_layer, n_units_per_layer), torch.nn.Tanh()] * (n_hidden_layers-1))
        self.phi_layers.extend([torch.nn.Linear(n_units_per_layer, n_latent_dims), torch.nn.Tanh()])
        self.phi = torch.nn.Sequential(*self.phi_layers)

        self.action_head_layers = []
        self.action_head_layers.extend([torch.nn.Linear(2 * n_latent_dims, n_units_per_layer), torch.nn.Tanh()])
        self.action_head_layers.extend([torch.nn.Linear(n_units_per_layer, n_units_per_layer), torch.nn.Tanh()] * (n_hidden_layers-1))
        self.action_head_layers.extend([torch.nn.Linear(n_units_per_layer, self.n_actions)])
        self.action_head = torch.nn.Sequential(*self.action_head_layers)

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.softmax = torch.nn.Softmax(dim=-1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def layers(self):
        phi_sequential = list(self.phi.modules())[0]
        phi_layers = list(phi_sequential.modules())[1:]
        action_head_sequential = list(self.phi.modules())[0]
        action_head_layers = list(action_head_sequential.modules())[1:]
        return phi_layers + action_head_layers

    def forward(self, x0, x1):
        z0 = self.phi(x0)
        z1 = self.phi(x1)
        context = torch.cat((z0,z1), -1)
        a_logits = self.action_head(context)
        return a_logits

    def predict_a(self, x0, x1):
        a_logits = self(x0, x1)
        return torch.argmax(a_logits, dim=-1)

    def compute_loss(self, x0, x1, a):
        a_logits = self(x0, x1)
        loss = self.cross_entropy(input=a_logits, target=a)
        return loss

    def train_batch(self, x0, x1, a):
        self.optimizer.zero_grad()
        loss = self.compute_loss(x0, x1, a)
        loss.backward()
        self.optimizer.step()
        return loss
