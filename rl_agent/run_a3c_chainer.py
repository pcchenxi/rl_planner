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

# from chainerrl.agents import a3c
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl import v_function

import sys, os
sys.path.append("../environment") 
import env_vrep
import a3c
import simplejson
import time
file_name = time.time()
processor_status = np.zeros(50)

f = open("../data/hist.txt", "w") 
f.close()

def save_model(env, agent, global_t):
    if global_t%20 == 0:
        chainer.serializers.save_npz("../model/" + str(file_name) + ".model", agent.model)
    # if global_t%400 == 0:
    #     file_name = time.time()
    if global_t%agent.t_max == 0 and global_t > 50:
        f = open("../data/hist.txt", "a") 
        f.write("%s %s %s %s\n" % (global_t, agent.pi_loss[0], agent.v_loss[0][0], agent.total_loss[0]))
        f.close()

def phi(obs):
    return obs.astype(np.float32)

class A3CFFSoftmax_laser(chainer.ChainList, a3c.A3CModel):
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
            self.v=L.Linear(128, 1)

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

class A3CFFSoftmax_basic(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(30, 30)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes = hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        path = state[:, :2]
        laser = state[:, 2:]
        return self.pi(path), self.v(path)


class A3CFFMellowmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward mellowmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.MellowmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CLSTMGaussian(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):
    """An example of A3C recurrent Gaussian policy."""

    def __init__(self, obs_size, action_size, hidden_size=200, lstm_size=128):
        self.pi_head = L.Linear(obs_size, hidden_size)
        self.v_head = L.Linear(obs_size, hidden_size)
        self.pi_lstm = L.LSTM(hidden_size, lstm_size)
        self.v_lstm = L.LSTM(hidden_size, lstm_size)
        self.pi = policies.LinearGaussianPolicyWithDiagonalCovariance(
            lstm_size, action_size)
        self.v = v_function.FCVFunction(lstm_size)
        super().__init__(self.pi_head, self.v_head,
                         self.pi_lstm, self.v_lstm, self.pi, self.v)

    def pi_and_v(self, state):

        def forward(head, lstm, tail):
            h = F.relu(head(state))
            h = lstm(h)
            return tail(h)

        pout = forward(self.pi_head, self.pi_lstm, self.pi)
        vout = forward(self.v_head, self.v_lstm, self.v)

        return pout, vout


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--arch', type=str, default='FFSoftmax',
                        choices=('FFSoftmax', 'FFMellowmax', 'LSTMGaussian'))
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-1)
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
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

    model = A3CFFSoftmax_basic(2, action_space)
    # model = A3CFFSoftmax_laser(obs_space, action_space)

    # chainer.serializers.load_npz("../model/xr.model", model)

    opt = rmsprop_async.RMSpropAsync(
        lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)
    opt.setup(model)
    # opt.add_hook(chainer.optimizer.GradientClipping(40))
    # if args.weight_decay > 0:
    #     opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    agent = a3c.A3C(model, opt, t_max=args.t_max, gamma=0.99,
                    beta=args.beta, phi=phi)
    if args.load:
        agent.load(args.load)


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