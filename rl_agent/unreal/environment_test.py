# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

from environment.environment import Environment


class TestEnvironment(unittest.TestCase):
    def test_gym(self):
        env_type = "vrep"
        env_name = "vrep_env"
        self.check_environment(env_type, env_name)

    def check_environment(self, env_type, env_name):
        env = Environment.create_environment(env_type, env_name, 0)
        action_size = Environment.get_action_size(env_type, env_name)

        for i in range(3):
            state, reward, terminal = env.process(0)

            print (state)
            print (reward)
            print (terminal)
            # # Check shape
            # self.assertTrue(state.shape == (84, 84, 3))
            # # state and pixel_change value range should be [0,1]
            # self.assertTrue(np.amax(state) <= 1.0)


        env.stop()



if __name__ == '__main__':
    unittest.main()
