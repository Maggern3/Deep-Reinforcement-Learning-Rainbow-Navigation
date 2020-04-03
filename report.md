

## At the end of the rainbow

The algorithm implemented is Distributional Double Dueling Deep Q-Network /w multi step bootstrap targets, prioritized experience replay 
and noisy linear layers aka rainbow.

report.txt describes the different variation of hyperparameters I've tried out and their results. For brevity I've included the following in this report.
The others are mostly standard rainbow values from the relevant papers.

buffer_size = 100000

Start training after 10000 frames have been put in the buffer.

Batch size of 32 SARS tuples per learning step.

The neural networks used has two fully connected noisy layers with 512 weights for each input channel. The advantage stream network has 4 actions multiplied with the number of atoms(51).
The state value stream only outputs the atoms(51).

### Results
The environment was not solved in 3000 episodes, 900.000 frames of learning.

[image1]: https://github.com/Maggern3/Deep-Reinforcement-Learning-Rainbow-Navigation/blob/master/v7.png "Trained Agent v7"

![Trained Agent v7][image1]

### Ideas for future work
Adjusting the sampling code to account for terminal state's in the view of multi step targeting.
