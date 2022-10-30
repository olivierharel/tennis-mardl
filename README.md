[//]: # (Image References)

[image1]: https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png "Trained Agent"

# Tennis game project

### Introduction

This project uses the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment from Unity adapted by Udacity.

![Unity ML-Agents Tennis Environment][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

### Multi-Agent RL training with MADDPG

We use an implementation of Multi-Agent Deep Deterministic Policy Gradient (MADDPG) based on the Open-AI paper: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf). In the case of the tennis game,
we leverage the competitive interaction between the 2 agents in a shared environment to learn a single set of weights.

Without clear documentation on how the observations by the 2 agents differ, we use a small 'normalizer' network to normalize the state across the agents, before passing the normalized state to the common actor and critic. This small network takes in the id of the play (0 or 1) and learns a set of 8 factors (per agent) that are multiplied with the input state. A specific loss term is added for the normalizer that constraints it generate factors close to 1 in absolute value (that is either +1 or -1). This approach could be generalized to environments where the observations by multiple agents differ in a more complex manner.

The task is episodic, and the environment is considered solved once the agents get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

###  Setup the project in your environment

1. Dependencies
   Setup the python environment following https://github.com/udacity/deep-reinforcement-learning#dependencies

2. Download the Unity environment from one of the links below. 
   You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

3. Place the file in the local "Tennis-mardl" folder, and unzip it. 

### Instructions

a. There are 3 main program files:
- `Tennis.ipynb`: the notebook sets up the environment and orchestrates the training
- `maddpg.py`: implements the the maddpg class and sets some fundamental hyperparameters (network dimensions, training hyperparameters...)
- `model.py`: the DL networks used for the local/target critic/actor/normalizer

b. `Continuous_Control.ipynb` has 4 sections:
  1. `Setup the general environment`: load the unity environment
  2. `Observe the environment space`: printout fundamental information about the environment (size of state/action spaces...)
  3. `Take random actions in the environment`: observe untrained agent
  4. `It's your turn`: learn an effective policy with MADDPG
  5. `Replay of trained agent`

b. To learn, execute steps 1, 2 and 4.
c. To replay a trained agent, execute steps 1, 2 and 5.

d. The weights are saved as:
- normalizer: `checkpoint_normalizer.pth`
- actor: `checkpoint_actor.pth`
- critic: `checkpoint_critic.pth`
(trained weights are uploaded on github)

