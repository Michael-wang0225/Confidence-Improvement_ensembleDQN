import torch
import torch.nn as nn
import torch.nn.functional as F





class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seedlayer
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ModelWithPrior(QNetwork):
    def __init__(self,
                 base_model : QNetwork,
                 prior_model : QNetwork,
                 state_size, action_size, seed, fc1_units=128, fc2_units=64,
                 prior_scale : float = 1.0):
        super().__init__()
        self.base_model = base_model
        self.prior_model = prior_model
        self.prior_scale = prior_scale

    def forward(self, inputs):
        with torch.no_grad():
            prior_out = self.prior_model(inputs)
            prior_out = prior_out.detach()
        model_out = self.base_model(inputs)
        return model_out + (self.prior_scale * prior_out)