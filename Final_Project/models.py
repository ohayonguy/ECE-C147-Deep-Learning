import torch
import torch.nn as nn
import torch.nn.functional as F

activations = {
    'relu': F.relu,
    'leakyrelu' : F.leaky_relu,
    'tanh': F.tanh,
    'none': lambda x: x,
}


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size, batch_norm=False, dropout=False, activation='none', init=('const', 1)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        self.seed = torch.manual_seed(seed)
        super(QNetwork, self).__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.affine1 = nn.Linear(state_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.affine2 = nn.Linear(hidden_size, action_size)
        with torch.no_grad():
            if init[0] == 'uniform':
                torch.nn.init.uniform_(self.affine1.weight, a=-init[1], b=init[1])
                torch.nn.init.uniform_(self.affine2.weight, a=-init[1], b=init[1])
            elif init[0] == 'const':
                torch.nn.init.constant_(self.affine1.weight, init[1])
                torch.nn.init.constant_(self.affine2.weight, init[1])
            elif init[0] == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(self.affine1.weight)
                torch.nn.init.xavier_uniform_(self.affine2.weight)
            elif init[0] == 'pytorch_default':
                pass
            else:
                raise Exception("Unsupported initialization")

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.affine1(state)
        x = activations[self.activation](x)
        if self.dropout:
            x = F.dropout(x, p=0.5)
        if self.batch_norm:
            x = self.bn(x)
        action_scores = self.affine2(x)
        return action_scores


class Policy(QNetwork):
    def __init__(self, state_size, action_size, seed, hidden_size, batch_norm=False, dropout=False, activation='none', init=('const', 1)):
        super(Policy, self).__init__(state_size, action_size, seed, hidden_size, batch_norm, dropout, activation, init)

    def forward(self, state):
        return F.softmax(super(Policy, self).forward(state), dim=1)
