from unityagents import UnityEnvironment
import numpy as np
import torch

env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
observation_state_size = brain.vector_observation_space_size
action_space_size = brain.vector_action_space_size

epsilon = 0
eps_decay = 0.99
eps_min = 0.001
gamma = 0.99
training_interval = 4

from dqnagent import DQNAgent
agent = DQNAgent(observation_state_size, action_space_size)
agent.network1.load_state_dict(torch.load('checkpoint.pth'))
for episode in range(0, 2):
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]     
       # get the current state
    score = 0                                          # initialize the score
    while(True):
        action = agent.select_action(state, 0)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                              # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
        
    print("Episode {} Score: {} epsilon {}".format(episode, score, epsilon))


    #todo list fixes:   
    

    #performance improvements
    #increase network depth
    # lr, 0.0001?
    # batch_select