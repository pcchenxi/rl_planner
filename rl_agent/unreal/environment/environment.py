# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# import sys
# print (sys.path)

class Environment(object):
  # cached action size
    action_size = -1

    @staticmethod
    def create_environment(env_type, env_name, process_idx):
        from . import vrep_environment
        return vrep_environment.VrepEnvironment('vrep_env', process_idx)

    @staticmethod
    def get_action_size(env_type, env_name):
        if Environment.action_size >= 0:
            return Environment.action_size

        print ('evn test', env_type)
        from . import vrep_environment
        Environment.action_size = vrep_environment.VrepEnvironment.get_action_size()

        return Environment.action_size

    def __init__(self):
      pass

    def process(self, action):
      pass

    def reset(self):
      pass

    def stop(self):
      pass  
