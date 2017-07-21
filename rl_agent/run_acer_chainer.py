"""An example of training A3C against OpenAI Gym Envs.
This script is an example of training a PCL agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.
To solve CartPole-v0, run:
    python train_a3c_gym.py 8 --env CartPole-v0
To solve InvertedPendulum-v1, run:
    python train_a3c_gym.py 8 --env InvertedPendulum-v1 --arch LSTMGaussian --t-max 50  # noqa
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import argparse

import chainer
from chainer import functions as F
from chainer import links as L
import gym
import gym.wrappers
import numpy as np

from chainerrl.action_value import DiscreteActionValue
# from chainerrl.agents import acer
from chainerrl.distribution import SoftmaxDistribution
from chainerrl import experiments
from chainerrl.initializers import LeCunNormal
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl import q_functions
from chainerrl.replay_buffer import EpisodicReplayBuffer
from chainerrl import spaces
from chainerrl import v_functions

import sys, os
sys.path.append("../environment") 
import env_vrep

import matplotlib.pyplot as plt
from drawnow import drawnow
import acer

import time
file_name = time.time()
processor_status = np.zeros(50)


def save_model(env, agent, global_t):
    if global_t%20 == 0:
        chainer.serializers.save_npz("../model/" + str(file_name) + ".model", agent.model)
    # if global_t%400 == 0:
    #     file_name = time.time()
    # if global_t%agent.t_max == 0:
    #     y.append(agent.total_loss)
    #     x.append(global_t)  # or any arbitrary update to your figure's data
    #     drawnow(make_fig)


def phi(obs):
    return obs.astype(np.float32)

class A3CFFSoftmax_laser(chainer.ChainList, acer.ACERSeparateModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, obs_size, n_actions):
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=1, out_channels=16, ksize=[1, 8], stride=1, pad=[0, 4])
            self.conv2 = L.Convolution2D(in_channels=16, out_channels=32, ksize=[1, 4], stride=1, pad=[0, 2])
            self.conv3 = L.Convolution2D(in_channels=32, out_channels=32, ksize=[1, 4], stride=1, pad=[0, 2])
            self.l_path = links.MLP(2, 30, hidden_sizes=(30,30))
            self.l1=L.Linear(5856, 64)
            self.l2=L.Linear(94, 256)
            self.l3=L.Linear(256, 29) #actor
            self.l4=L.Linear(256, 128) # critic
            self.pi=policies.SoftmaxPolicy(model = L.Linear(29, n_actions))
            self.v=L.Linear(128, n_actions)

        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        path = state[:, :2]
        laser = state[:, 2:]

        # laser net
        laser_in = np.expand_dims(laser, axis=1)
        laser_in = np.expand_dims(laser_in, axis=1)

        h = F.relu(self.conv1(laser_in))
        # print (h.shape)
        # h = F.max_pooling_2d(h, 2, 2)
        # print (h.shape)
        h = F.relu(self.conv2(h))
        # print (h.shape)
        # h = F.max_pooling_2d(h, 2, 2)
        # print (h.shape)
        h = F.relu(self.conv3(h))
        # print (h.shape) 
        flat = h.data

        fc1_in = np.zeros([flat.shape[0], flat.shape[1]*flat.shape[2]*flat.shape[3]]).astype(np.float32)
        for i in range(flat.shape[0]):
            temp = flat[i]
            temp = temp.flatten()
            fc1_in[i] = temp

        # print (fc1_in.shape)
        h = F.relu(self.l1(fc1_in))

        # path net
        path = self.l_path(path)
        # print (path.shape)

        ## add target position
        flat = h.data
        fc2_in = np.zeros([flat.shape[0], flat.shape[1]+path.shape[1]]).astype(np.float32)

        for i in range(flat.shape[0]):
            temp2 = np.append(flat[i], path[i].data)
            fc2_in[i] = temp2

        features = F.relu(self.l2(fc2_in))

        value_pi = F.relu(self.l3(features))
        value_v = F.relu(self.l4(features))

        value_pi = self.pi(value_pi)
        value_v = self.v(value_v)
        # print fc2_in.shape

        # h = self.l3(h)

        return value_pi, value_v


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--t-max', type=int, default=50)
    parser.add_argument('--n-times-replay', type=int, default=4)
    parser.add_argument('--n-hidden-channels', type=int, default=30)
    parser.add_argument('--n-hidden-layers', type=int, default=2)
    parser.add_argument('--replay-capacity', type=int, default=5000)
    parser.add_argument('--replay-start-size', type=int, default=10 ** 3)
    parser.add_argument('--disable-online-update', action='store_true')
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-2)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--truncation-threshold', type=float, default=5)
    parser.add_argument('--trust-region-delta', type=float, default=0.1)
    args = parser.parse_args()

    logging.getLogger().setLevel(args.logger_level)

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        if test:
            return []
        print ('in make env', process_idx, test)
        # env = gym.make(args.env)
        env = []
        if processor_status[process_idx] == 0:
            env = env_vrep.Simu_env(20000 + process_idx)
            env.connect_vrep()
            processor_status[process_idx] = 1
        return env

    # sample_env = gym.make(args.env)
    # timestep_limit = sample_env.spec.tags.get(
    #     'wrapper_config.TimeLimit.max_episode_steps')
    # obs_space = sample_env.observation_space
    # action_space = sample_env.action_space
    obs_space = env_vrep.state_size
    action_space = env_vrep.action_size
    timestep_limit = 200

    pi = policies.FCSoftmaxPolicy( obs_space, action_space, n_hidden_channels=args.n_hidden_channels,n_hidden_layers=args.n_hidden_layers)
    v = v_functions.FCVFunction( obs_space,n_hidden_channels=args.n_hidden_channels,n_hidden_layers=args.n_hidden_layers)
    adv = q_functions.FCSAQFunction(obs_space, action_space,n_hidden_channels=args.n_hidden_channels, n_hidden_layers=args.n_hidden_layers)

    model = acer.ACERSDNSeparateModel(pi, v, adv)
    # model = acer.ACERSeparateModel(
    #     pi=links.Sequence(
    #         L.Linear(obs_space, args.n_hidden_channels),
    #         F.relu,
    #         L.Linear(args.n_hidden_channels, action_space,
    #                     initialW=LeCunNormal(1e-3)),
    #         SoftmaxDistribution),
    #     q=links.Sequence(
    #         L.Linear(obs_space, args.n_hidden_channels),
    #         F.relu,
    #         L.Linear(args.n_hidden_channels, action_space,
    #                     initialW=LeCunNormal(1e-3)),
    #         DiscreteActionValue),
    # )

    opt = rmsprop_async.RMSpropAsync(
        lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))

    replay_buffer = EpisodicReplayBuffer(args.replay_capacity)
    agent = acer.ACER(model, opt, t_max=args.t_max, gamma=0.99,
                      replay_buffer=replay_buffer,
                      n_times_replay=args.n_times_replay,
                      replay_start_size=args.replay_start_size,
                      disable_online_update=args.disable_online_update,
                      use_trust_region=True,
                      trust_region_delta=args.trust_region_delta,
                      truncation_threshold=args.truncation_threshold,
                      beta=args.beta, phi=phi)

    experiments.train_agent_async(
        agent=agent,
        outdir=args.outdir,
        processes=args.processes,
        make_env=make_env,
        profile=args.profile,
        steps=args.steps,
        eval_n_runs=args.eval_n_runs,
        eval_interval=args.eval_interval,
        max_episode_len=timestep_limit,
        global_step_hooks = [save_model])

if __name__ == '__main__':
    main()