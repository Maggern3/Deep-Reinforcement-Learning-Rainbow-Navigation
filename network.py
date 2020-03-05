import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):

    def __init__(self, observation_state_size, action_space_size):
        super(DeepQNetwork, self).__init__()
        self.seed = torch.manual_seed(0)
        self.fc1 = nn.Linear(observation_state_size, 256)
        self.fc2 = nn.Linear(256, action_space_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x