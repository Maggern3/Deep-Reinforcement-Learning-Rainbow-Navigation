from unityagents import UnityEnvironment
import numpy as np
import torch
import time

env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
observation_state_size = brain.vector_observation_space_size
action_space_size = brain.vector_action_space_size

epsilon = 1.0
eps_decay = 0.995
eps_min = 0.01
gamma = 0.99
training_interval = 4
n_atoms = 51
v_min = -10
v_max = 10
sampling_beta = 0.4
sampling_beta_eps = (1 - sampling_beta) / 1000
buffer_size = 10000

from dqnagent import DQNAgent
agent = DQNAgent(observation_state_size, action_space_size, n_atoms, v_min, v_max, buffer_size)

t=0
for episode in range(0, 1000):
    start = time.time()
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]     
       # get the current state
    score = 0                                          # initialize the score
    epsilon = max(epsilon*eps_decay, eps_min)  
    sampling_beta = min(sampling_beta+sampling_beta_eps, 1.0)
    while(True):
        t+=1
        action = agent.select_action(state, epsilon)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        sars = (state, action, reward, next_state, done)
        agent.add(sars)
        if(t % training_interval==0):            
            agent.train_from_samples_distributional_dqn(gamma, n_atoms, v_min, v_max, sampling_beta)            
            #print('training round took {:.1f} sec'.format(end - start))
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
    end = time.time()
    print("Episode {} Score: {} frame {} took {:.1f} sec sampling_beta {:.3f} epsilon {:.4f}".format(episode, score, t, end - start, sampling_beta, epsilon))
torch.save(agent.network1.state_dict(), 'checkpoint5.pth')

    #add running mean scores

    #performance improvements
    # increase network depth
    # lr, 0.0001?    
    # increase buffer size! 100.000 for lunar lander
    # dueling
    # noisy
    # modify bootstrap targeting to include current reward?
    # change to log softmax
