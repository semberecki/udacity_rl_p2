import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Policy(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

        #dist = torch.distributions.Normal(action, std)

    # def __init__(self):
    #     super(Policy, self).__init__()
    #     # 80x80x2 to 38x38x4
    #     # 2 channel from the stacked frame
    #     self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
    #     # 38x38x4 to 9x9x32
    #     self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
    #     self.size=9*9*16
    #
    #     # two fully connected layer
    #     self.fc1 = nn.Linear(self.size, 256)
    #     self.fc2 = nn.Linear(256, 1)
    #
    #     # Sigmoid to
    #     self.sig = nn.Sigmoid()
    #
    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #     x = x.view(-1,self.size)
    #     x = F.relu(self.fc1(x))
    #     return self.sig(self.fc2(x))
