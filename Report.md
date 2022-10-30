[//]: # (Image References)

[image2]: https://github.com/olivierharel/tennis-mardl/blob/main/Training_curve.png "Training score curve"
[image3]: https://github.com/olivierharel/tennis-mardl/blob/main/trained_agent.webp "Trained agent"
[image4]: https://github.com/olivierharel/tennis-mardl/blob/main/untrained_agent.webp "Untrained agent"

# Continuous Control project (Reacher) report

### Hyper parameters

The code uses the following hyper parameters.

1. train_maddpg() parameters:
**max_t** sets the maximum number of timesteps per episodes.
**update_period** and **training_per_update** set the relative frequency of training phases and data gathering phases:
- At each timestep, in an episode 2 experiences (one per player) are gathered and added to the replay buffer
- update_period specifies the period, in timesteps, between training phases of the actor-critic architecture
- training_per_update specifies the number fo training iterations within a training phase

2. maddpg class parameters:
- Replay buffer size: training starts with a (smaller) **buffer_size_init** capacity (1e5) which is switched to 
  a (larger) **buffer_size_final** capacity (1e6) has soon as a certain episode score has been achieved.
  Why? At the beginning, with little training, the value of the older experiences is low. Also episodes are
  short and a smaller buffer can fit experiences from many episodes. Once a certain level of training has been
  achieved, episodes are much longer and a larger buffer is used to maintain a larger capacity in #episodes. This may 
  help stabilize the agents and avoid situations where agents could forget their training.
- **gamma**: discount factor = 0.99
- Epsilon, the noise scaling factor: **epsilon_init** (1.0) is the initial factor, it is decayed by **epsilon_decay** (0.99)
  on a per episode basis down to a minimum value **epsilon_min** (1e-2)
- Tau, the soft update factor: **tau_init** (1e-2) is the initial factor, it is decayed by **tau_decay** (0.985)
  on a per episode basis, where the episode score is high enough, down to a minimum value **tau_min** (1e-3)
  The point is to stabilize the agents once a certain level of training has been achieved.

2. Training parameters in are listed at the top of ddpg_agent_ac.py:
- **BUFFER_SIZE** = int(1e5)  # replay buffer size
- **BATCH_SIZE** = 128        # minibatch size
- **LR_NORMALIZER** = 1e-3    # learning rate of the normalizer
- **LR_ACTOR** = 1e-3         # learning rate of the actor 
- **LR_CRITIC** = 1e-3        # learning rate of the critic
- **WEIGHT_DECAY** = 1e-5     # L2 weight decay
- **ACT_FC1** = 200           # Hidden size of the actor network (layer 1)
- **ACT_FC2** = 100           # Hidden size of the actor network (layer 2)
- **CRI_FC1** = 200           # Hidden size of the critic network (layer 1)
- **CRI_FC2** = 128           # Hidden size of the critic network (layer 2)
- **CRI_FC3** = 128           # Hidden size of the critic network (layer 3)

### Training observations

Some key choices and tuning parameters did impact the success and stability of training more than others.
1. I struggled to figure out the difference of perspective between the 2 player agents (how the state variables differ in their
observations). I elected to let a simple network figure it out instead by using the 'normalizer' (see Readme).

2. For a while the agents wouldn't train: I was decaying epsilon too fast and the agents were simply not exposed to enough
useful experience to be able to learn. Slowing doing the decay jumpstarted training.

3. I did observe initially a 'learning' crash - where the agents would learn to play very well and then would suddenly
loose their skills through additional episodes, never to recover them again. I bumped up the replay buffer size and introduced 
the tau decay, as well as a minimum epsilon value (always inject at least a small amount of noise) to eliminate the issue successfully (that is maintain the average score for 1000 episodes after the average score has cleared 1.0).

With the committed settings, training was fast and stable:
![Training score curve][image2]

Side-by-side comparison of untrained and trained agents:
| Untrained | Trained |
|:---------:|:-------:|
| ![Untrained][image4] | ![Trained][image3] |

### Future work
- Try different hyper parameters
- Try more complex games (soccer) with even more agents
