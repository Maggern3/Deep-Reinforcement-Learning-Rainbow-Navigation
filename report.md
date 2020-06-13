

## Deep Q-Network

The algorithm implemented is Double Deep Q-Network

The algorithm uses neural networks to estimate q-values(state-action values) for large continous state spaces. The original DQN algorithm had several weaknesses. Each of the additions to DQN solves one of these problems.

Double DQN(dobule q-learning) solves overestimation of Q values while maintaining fast training times.

In the experience replay buffer it stores 10.000 experience samples.

It starts training after 64 frames have been put in the replay buffer.

The batch size is 64 SARS tuples per learning step.

The neural networks used has two fully connected layers with 256 weights for each input channel.

### Results
The environment was solved in 512 episodes.

[image1]: https://github.com/Maggern3/Rainbow/blob/master/results.png "Trained Agent"

![Trained Agent][image1]

### Ideas for future work
Decay exploration based on number of episodes, bigger neural network, bigger experience buffer, delay start of training. Prioritized experience buffer.
