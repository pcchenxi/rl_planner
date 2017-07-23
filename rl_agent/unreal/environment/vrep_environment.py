# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import numpy as np

from environment.environment import Environment
from environment import env_vrep
import time

COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2


def preprocess_frame(observation):

    return observation

def worker(conn, env_name, process_idx = 0):
    env = env_vrep.Simu_env(20000 + process_idx)
    env.connect_vrep()
    env.reset()
    conn.send(0)

    while True:
        command, arg = conn.recv()
        # print (command)

        if command == COMMAND_RESET:
            print ('reset')
            obs = env.reset()
            state = preprocess_frame(obs)
            conn.send(state)
        elif command == COMMAND_ACTION:
            reward = 0
            for i in range(4):
                obs, r, terminal, _ = env.step(arg)
                # print (r)
                reward += r
                if terminal:
                    break
            state = preprocess_frame(obs)
            conn.send([state, reward, terminal])
        elif command == COMMAND_TERMINATE:
            print ('terminate')
            break
        else:
            print("bad command: {}".format(command))
    env.disconnect_vrep()
    conn.send(0)
    conn.close()


class VrepEnvironment(Environment):
    @staticmethod
    def get_action_size():
        action_size = env_vrep.action_size
        return action_size

    def __init__(self, env_name, process_idx):
        Environment.__init__(self)

        self.last_state = []
        self.last_action = []
        self.last_reward = []

        self.conn, child_conn = Pipe()
        self.proc = Process(target=worker, args=(child_conn, env_name, process_idx))
        self.proc.start()
        self.conn.recv()
        self.reset()

    def reset(self):
        self.conn.send([COMMAND_RESET, 0])
        self.last_state = self.conn.recv()

        self.last_action = 0
        self.last_reward = 0

    def stop(self):
        self.conn.send([COMMAND_TERMINATE, 0])
        ret = self.conn.recv()
        self.conn.close()
        self.proc.join()
        print("vrep environment stopped")

    def process(self, action):
        self.conn.send([COMMAND_ACTION, action])
        time.sleep(0.1)
        state, reward, terminal = self.conn.recv()

        self.last_state = state
        self.last_action = action
        self.last_reward = reward
        return state, reward, terminal
