# Deep-Reinforcement-Learning-Rainbow-Navigation

The state space consists of 37 different inputs with ray-based vision of objects and the velocity of the agent.

The agent has 4 available actions, moving forward, backwards and turning left or right.

The goal is to collect as many yellow bananas as possible and to avoid the blue bananas.

The environment is considered solved when your agent averages scores above 13 for 100 episodes. One episode lasts for 300 frames.

### How to run the project
First clone the [udacity deep reinforcement learning repo](https://github.com/udacity/deep-reinforcement-learning) 
and navigate to it's directory then
```
cd python
pip install .
```
this installs the required dependencies. 

Download the Banana world unity ml environment from one of the following links:

Windows: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
     
Mac: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
     
Put it in the root of the cloned project folder and unzip the file. 

You should now be able to build and run the project.

### How to train the agent
Run the following command to train the agent
```
python runner.py
```
