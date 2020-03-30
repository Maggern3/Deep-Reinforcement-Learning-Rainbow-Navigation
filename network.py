import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):

    def __init__(self, observation_state_size, action_space_size, n_atoms):
        super(DeepQNetwork, self).__init__()
        self.seed = torch.manual_seed(0)
        self.fc1 = nn.Linear(observation_state_size, 256)
        self.fc2 = nn.Linear(256, action_space_size * n_atoms)
        self.fc1state = nn.Linear(observation_state_size, 256)
        self.fc2state = nn.Linear(256, n_atoms)
        self.action_space_size = action_space_size
        self.n_atoms = n_atoms

    def forward(self, state):
        a_v = F.relu(self.fc1(state))
        a_v = self.fc2(a_v)
        s_v = F.relu(self.fc1state(state))
        s_v = self.fc2state(s_v)
        s_v = s_v.view(-1, 1, self.n_atoms)
        a_v = a_v.view(-1, self.action_space_size, self.n_atoms)
        x = s_v +  a_v - a_v.mean(1, keepdim=True)
        return F.softmax(x, dim=2)