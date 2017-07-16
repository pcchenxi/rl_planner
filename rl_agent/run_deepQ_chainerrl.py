import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np

import time

import sys, os
sys.path.append("../environment") 

import env_vrep

file_name = time.time()

# init simu environment
env = env_vrep.Simu_env(10000)
env.connect_vrep(True)

class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super(QFunction, self).__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            # l1=L.Linear(n_hidden_channels, n_hidden_channels),
            # l2=L.Linear(n_hidden_channels, n_hidden_channels),
            # l3=L.Linear(n_hidden_channels, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_actions))

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        h = F.relu(self.l0(x))
        # h = F.relu(self.l1(h))
        # h = F.relu(self.l2(h))
        # h = F.relu(self.l3(h))

        # h = F.tanh(self.l1(h))
        # print x
        # print self.l1(h)
        # print chainerrl.action_value.DiscreteActionValue(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l1(h))

def sample_action():
    # print 'in action'
    action = np.random.randint(0, n_actions)
    return action


obs_size = env.state_size
n_actions = env.action_size
q_func = QFunction(obs_size, n_actions)


# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

# Set the discount factor that discounts future rewards.
gamma = 0.95

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.2, random_action_func=sample_action)
# explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
#     start_epsilon = 1, end_epsilon = 0.2, decay_steps = 3000, random_action_func=sample_action)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
agent = chainerrl.agents.DQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=40, update_interval=1,
    target_update_interval=100, phi=phi)

n_episodes = 20000
max_episode_len = 200

chainer.serializers.load_npz("../model/deepq_xy.model", q_func)

for i in range(1, n_episodes + 1):
    obs = env.reset()
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while not done and t < max_episode_len:
        # Uncomment to watch the behaviour
        # env.render()
        action = agent.act_and_train(obs, reward)
        # print action
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        print obs, action, reward
        o = obs.reshape(1, -1)
        o = o.astype(np.float32, copy=False)
        print q_func(o)
        if t % 10 == 0:
            chainer.serializers.save_npz("../model/" + str(file_name) + ".model", q_func)

    # if i % 10 == 0:
    print('episode:', i,
            'R:', R,
            'statistics:', agent.get_statistics())
    if i % 100 == 0:
        file_name = time.time()
    agent.stop_episode_and_train(obs, reward, done)
print('Finished.')


