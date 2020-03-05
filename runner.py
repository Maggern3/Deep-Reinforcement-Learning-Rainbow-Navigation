from unityagents import UnityEnvironment
import numpy as np
import torch

env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
observation_state_size = brain.vector_observation_space_size
action_space_size = brain.vector_action_space_size

epsilon = 1.0
eps_decay = 0.99
eps_min = 0.001
gamma = 0.99
training_interval = 4

from dqnagent import DQNAgent
agent = DQNAgent(observation_state_size, action_space_size)

for episode in range(0, 2000):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]     
       # get the current state
    score = 0                                          # initialize the score
    epsilon = max(epsilon*eps_decay, eps_min)
    while(True):
        action = agent.select_action(state, epsilon)
        #action = np.random.randint(4)        # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        sars = (state, action, reward, next_state, done)
        agent.add(sars)
        if(t % training_interval==0):
            agent.train_from_samples(gamma)
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
        
    print("Episode {} Score: {} epsilon {}".format(episode, score, epsilon))
torch.save(agent.network1.state_dict(), 'checkpoint3.pth')

    #add running mean scores

    #performance improvements
    # increase network depth
    # lr, 0.0001?    
    # increase buffer size
    # rainbow