import numpy as np
from collections import deque
from network import DeepQNetwork
import torch
import torch.optim as optim
import torch.nn.functional as F

class DQNAgent:
    # make two networks, to implement sgd
    def __init__(self, observation_state_size, action_space_size, n_atoms, v_min, v_max, buffer_size):
        gpu = torch.cuda.is_available()
        if(gpu):
            print('GPU/CUDA works! Happy fast training :)')
            torch.cuda.current_device()
            torch.cuda.empty_cache()
            self.device = torch.device("cuda")
        else:
            print('training on cpu...')
            self.device = torch.device("cpu")
        self.network1 = DeepQNetwork(observation_state_size, action_space_size, n_atoms).to(self.device)
        self.fixednetwork = DeepQNetwork(observation_state_size, action_space_size, n_atoms).to(self.device)# copy from network 1, hmmmm, copy all weights??
        self.action_space_size = action_space_size
        self.optim = optim.Adam(self.network1.parameters(), lr=0.0001)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = 32
        self.tau = 0.001
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = torch.Tensor(self.z_v(n_atoms, v_min, v_max))#.to(self.device) #or z_j
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n = 3
        self.sampling_alpha = 0.5
        self.training_start = 10000

    def z_v(self, n_atoms, v_min, v_max):
        z = []
        for i in range(1, n_atoms+1):
            z_i = v_min + (i-1)*((v_max - v_min)/(n_atoms - 1))
            z.append(z_i)
        return z

    def train_from_samples_distributional_dqn(self, gamma, n_atoms, v_min, v_max, sampling_beta):
        self.network1.reset_noise()
        self.fixednetwork.reset_noise()
        if(len(self.replay_buffer) < self.training_start):
            return
        optim = self.optim
        optim.zero_grad()
        indices, states, actions, rewards, next_states, dones, weights = self.select_samples(gamma, sampling_beta)

        with torch.no_grad():     
            p = self.fixednetwork(next_states).data.cpu() # torch.Size([64, 4, 51])        
            # atoms torch.Size([51]),
            #verified correct, multiplies each atom with each atom for each action in every batch
            q = p * self.atoms # torch.Size([64, 4, 51])
            online_q = self.network1(next_states) * self.atoms
            q_star_double = online_q.sum(2).max(1)[1] #torch.Size([64])      

            # rewards torch.Size([64, 1]), atoms torch.Size([51]), dones torch.Size([64, 1])
            t = rewards + (gamma ** self.n) * self.atoms * (1-dones) # torch.Size([64, 51])    
            t = t.clamp(self.v_min, self.v_max) 
            b = t - v_min / self.delta_z  # torch.Size([64, 51])        
            l = b.floor().long() # torch.Size([64, 51])   
            u = b.ceil().long() # torch.Size([64, 51])
            m = torch.zeros((self.batch_size, self.action_space_size, n_atoms))#.to(self.device) # torch.Size([64, 4, 51])
            for batch in range(self.batch_size):
                for atom in range(n_atoms):
                    if((u[batch, atom] - b[batch, atom]) == 0):
                        u[batch, atom] += 1
                        l[batch, atom] -= 1
                    #if((u[batch, atom] - b[batch, atom]) == 0):
                    #    print('disappearing probability mass b', b)
                    m[batch, q_star_double[batch], l[batch, atom]] += p[batch, q_star_double[batch], atom] * (u[batch, atom] - b[batch, atom])
                    m[batch, q_star_double[batch], u[batch, atom]] += p[batch, q_star_double[batch], atom] * (b[batch, atom] - l[batch, atom])                        
        
        #loss = −∑i mi log pi(xt,at)
        actions = actions.unsqueeze(1) # torch.Size([64, 1, 1])        
        actions = actions.expand(self.batch_size, 1, n_atoms) # torch.Size([64, 1, 51])    
        # get the atoms for state given action
        old_prob_dist = self.network1(states).gather(1, actions) # gathers values along an axis specified by dim.  torch.Size([64, 1, 51])
        #print(old_prob_dist[0])
        #old_prob_dist = old_prob_dist.squeeze(1) # torch.Size([64, 51])      
        m = m.to(self.device)
        p_loss = -(m * torch.log(old_prob_dist)).sum(1) #torch.Size([64, 4, 51]) before sum, torch.Size([64, 51]) after        
        #print(p_loss[0])
        loss = (weights* p_loss).mean()
        #print('loss ', loss)
        loss.backward()
        optim.step()
        self.save_probabilities(indices, p_loss.sum(1).detach().cpu().numpy())
        self.fixed_network_update()

    def fixed_network_update(self):
        # copy weights from network1
        for network1_param, fixed_param in zip(self.network1.parameters(), self.fixednetwork.parameters()):
            fixed_param.data.copy_(self.tau * network1_param.data + (1.0-self.tau) * fixed_param.data)

    def add(self, sars):
        self.replay_buffer.append(sars)        
        if(len(self.priorities)==0):
            max_p = 1
        else:
            max_p = np.max(self.priorities)
        self.priorities.append(max_p)

    def save_probabilities(self, indices, batch_p_loss):
        for idx, p_loss in zip(indices, batch_p_loss):
            self.priorities[idx] = p_loss ** self.sampling_alpha

    def select_samples(self, gamma, sampling_beta):
        samples = []
        #sampling_probability = (p_i ** self.sampling_alpha) / (sum p_k ** self.sampling_alpha)
        valid_buffer_length = len(self.replay_buffer)-self.n
        p_k = np.array(self.priorities)[0:valid_buffer_length]
        sampling_probabilities = p_k / p_k.sum()
        for i in range(0, self.batch_size):
            selection = np.random.choice(valid_buffer_length, p=sampling_probabilities)
            state, action, reward, next_state, done = self.replay_buffer[selection]
            weight = ((1/len(p_k))*(1/sampling_probabilities[selection])) ** sampling_beta
            reward = 0
            steps = []
            for k in range(0, self.n):                
                #print('idx ',selection+k+1)
                ns = self.replay_buffer[selection+k+1]
                if(ns[4]==1): # if the selected next n-state is the final state
                    steps.append(0)
                else:
                    steps.append(ns[2])  
            for k in range(0, self.n):
                reward += (gamma ** k) * steps[k]
            #print(reward)
            samples.append((selection, state, action, reward, next_state, done, weight))
        indices = [s[0] for s in samples]
        states = torch.tensor([s[1] for s in samples]).float().to(self.device)
        actions = torch.tensor([s[2] for s in samples]).long().unsqueeze(1).to(self.device)
        rewards = torch.tensor([s[3] for s in samples]).float().unsqueeze(1)#.to(self.device)
        next_states = torch.tensor([s[4] for s in samples]).float().to(self.device)
        dones = torch.tensor([s[5] for s in samples]).float().unsqueeze(1)#.to(self.device)
        weights = torch.tensor([s[6] for s in samples]).float().unsqueeze(1) 
        return indices, states, actions, rewards, next_states, dones, weights

    def select_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        #self.network1.eval()
        with torch.no_grad():
            p = self.network1(state).data.cpu()                
            q = p * self.atoms
            action = int(q.sum(2).max(1)[1]) #np.argmax(q_v.numpy(), axis=1))   
        #self.network1.train()                     
        return action

