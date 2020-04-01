import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    def __init__(self, observation_state_size, action_space_size, n_atoms):
        super(DeepQNetwork, self).__init__()
        self.seed = torch.manual_seed(0)
        self.fc1 = NoisyLinearLayer(observation_state_size, 512)
        self.fc2 = NoisyLinearLayer(512, action_space_size * n_atoms)
        self.fc1state = NoisyLinearLayer(observation_state_size, 512)
        self.fc2state = NoisyLinearLayer(512, n_atoms)
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

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc1state.reset_noise()
        self.fc2state.reset_noise()

class NoisyLinearLayer(nn.Module):
    def __init__(self, input_size, output_size, variance=0.1): #consider changing variance to 0.5 if running on GPU
        super(NoisyLinearLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.variance = variance
        self.weights_mean = nn.Parameter(torch.zeros(output_size, input_size))
        self.weights_variance = nn.Parameter(torch.zeros(output_size, input_size))
        self.register_buffer('epsilon_weights', torch.zeros(output_size, input_size))
        self.bias_mean = nn.Parameter(torch.zeros(output_size))
        self.bias_variance = nn.Parameter(torch.zeros(output_size))
        self.register_buffer('epsilon_bias', torch.zeros(output_size))
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mean_range = 1 / math.sqrt(self.input_size)
        self.weights_mean.data.uniform_(-mean_range, mean_range)
        self.weights_variance.data.fill_(self.variance / math.sqrt(self.input_size))
        self.bias_mean.data.uniform_(-mean_range, mean_range)
        self.bias_variance.data.fill_(self.variance / math.sqrt(self.output_size))

    def reset_noise(self):
        input_noise = self.make_noise(self.input_size)
        output_noise = self.make_noise(self.output_size)
        self.epsilon_weights.copy_(output_noise.ger(input_noise)) # Outer product of input and vector2. 
        self.epsilon_bias.copy_(output_noise)

    def make_noise(self, noise_size):
        noise = torch.randn(noise_size)
        return noise.sign().mul_(noise.abs().sqrt()) # run square root on noise including negative values

    def forward(self, state):
        if(self.training):
            weights = self.weights_mean + self.weights_variance * self.epsilon_weights
            bias = self.bias_mean + self.bias_variance * self.epsilon_bias
            return F.linear(state, weights, bias)       
        else:
            return F.linear(state, self.weights_mean, self.bias_mean)