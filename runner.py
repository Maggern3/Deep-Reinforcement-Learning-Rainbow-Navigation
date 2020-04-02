from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import time
import matplotlib.pyplot as plt
#%matplotlib inline

env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
observation_state_size = brain.vector_observation_space_size
action_space_size = brain.vector_action_space_size

eps_decay = 0.995
eps_min = 0.01
gamma = 0.99
training_interval = 4
n_atoms = 51
v_min = -10
v_max = 10
sampling_beta = 0.4
sampling_beta_eps = (1 - sampling_beta) / 1000
buffer_size = 100000

from dqnagent import DQNAgent
agent = DQNAgent(observation_state_size, action_space_size, n_atoms, v_min, v_max, buffer_size)
scores = []
last_hundred_scores = deque(maxlen=100)
frame=0
for episode in range(0, 3000):
    start = time.time()
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]     
       # get the current state
    score = 0                                          # initialize the score
    sampling_beta = min(sampling_beta+sampling_beta_eps, 1.0)
    while(True):
        frame+=1
        action = agent.select_action(state)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        sars = (state, action, reward, next_state, done)
        agent.add(sars)
        if(frame % training_interval==0):            
            agent.train_from_samples_distributional_dqn(gamma, n_atoms, v_min, v_max, sampling_beta)            
            #print('training round took {:.1f} sec'.format(end - start))
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
    end = time.time()
    scores.append(score)
    last_hundred_scores.append(score)
    print("Episode {} Score: {} mean score: {} frame {} took {:.1f} sec sampling_beta {:.3f}".format(episode, score, np.mean(last_hundred_scores), frame, end - start, sampling_beta))

torch.save(agent.network1.state_dict(), 'checkpoint5.pth')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

    #performance improvements
    # increase network depth
    # lower lr further, 0.0000625?     
    # change to log softmax?
    # reward clipping?

    # increase network size, 512
    # increase buffer size! 100.000 for lunar lander
    # modify bootstrap targeting to include current reward?
    # decrease batch_size?