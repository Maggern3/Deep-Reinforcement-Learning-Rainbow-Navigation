import numpy as np
from collections import deque
from network import DeepQNetwork
import torch
import torch.optim as optim
import torch.nn.functional as F

class DQNAgent:
    # make two networks, to implement sgd
    def __init__(self, observation_state_size, action_space_size):
        self.network1 = DeepQNetwork(observation_state_size, action_space_size)
        self.fixednetwork = DeepQNetwork(observation_state_size, action_space_size)# copy from network 1, hmmmm, copy all weights??
        self.action_space_size = action_space_size
        self.optim = optim.Adam(self.network1.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.fixed_network_update(1.0)

    def add(self, sars):
        self.replay_buffer.append(sars)

    def train_from_samples(self, gamma):
        if(len(self.replay_buffer) < self.batch_size):
            return
        states, actions, rewards, next_states, dones = self.select_samples()
        optim = self.optim
        optim.zero_grad()
        argmax_actions_from_network1 = self.network1(next_states).detach().max(1)[1].unsqueeze(1)
        fixed = self.fixednetwork(next_states).detach().gather(1, argmax_actions_from_network1)
        q = rewards + gamma * fixed * (1-dones)    
        old_values = self.network1(states).gather(1, actions)
        loss = F.mse_loss(old_values, q)
        loss.backward()
        optim.step()        
        self.fixed_network_update()

    def fixed_network_update(self, tau=0.001):
        # copy weights from network1
        for network1_param, fixed_param in zip(self.network1.parameters(), self.fixednetwork.parameters()):
            fixed_param.data.copy_(tau * network1_param.data + (1.0-tau) * fixed_param.data)

    def select_samples(self):
        samples = []
        for i in range(0, self.batch_size):
            selection = np.random.randint(len(self.replay_buffer))
            samples.append(self.replay_buffer[selection])
        # TODO implement desired shaping
        states = torch.tensor([s[0] for s in samples]).float()
        actions = torch.tensor([s[1] for s in samples]).long().unsqueeze(1)
        rewards = torch.tensor([s[2] for s in samples]).float().unsqueeze(1)
        next_states = torch.tensor([s[3] for s in samples]).float()
        dones = torch.tensor([s[4] for s in samples]).float().unsqueeze(1) 
        return states, actions, rewards, next_states, dones

    def select_action(self, state, epsilon):
        #epsilon greedy policy
        # random action or greedy action?        
        greedy = np.random.choice(np.arange(2), p=[1-epsilon, epsilon])
        if(greedy==0):
            state = torch.tensor(state).float().unsqueeze(0)
            self.network1.eval()
            with torch.no_grad():
                q_v = self.network1(state)
            self.network1.train()
            action = int(np.argmax(q_v.numpy()))            
        else:
            action = np.random.choice(np.arange(self.action_space_size))
        return action

