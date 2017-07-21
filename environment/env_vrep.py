#!/usr/bin/env python
import sys, os, math
sys.path.append("../v-rep_plugin") 
import numpy as np
import time
## v-rep
import vrep

import matplotlib.pyplot as plt

action_list = []
state_size = 182
action_size = 9

vrep.simxFinish(-1) # just in case, close all opened connections
print 'init vrep'

for a in range(-1, 2):
    for b in range(-1, 2):
        # for c in range(-1, 2):
            # for d in range(-1, 2):
            #     for e in range(-1, 2):
        action = []
        action.append(0)
        action.append(b)
        action.append(a)
        action.append(0)
        action.append(0)
        # print action
        action_list.append(action)
        # print action_list

# print action_list


class Simu_env:
    def __init__(self, port_num):
        # super(Vrep_env, self).__init__(port_num)
        self.action_space = ['l', 'f', 'r', 'h', 'e']
        self.n_actions = len(self.action_space)
        # self.n_features = 2
        # self.title('Vrep_env')
        self.port_num = port_num
        self.reached_index = -1
        self.dist_pre = 100
        
        self.path_used = 1
        self.state_size = state_size
        self.action_size = action_size
        # self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        # self.clientID = self._connect_vrep(port_num)
        

    def connect_vrep(self, close_all = False):
        if close_all:
            vrep.simxFinish(-1) # just in case, close all opened connections

        clientID = vrep.simxStart('127.0.0.1',self.port_num,True,True,5000,5)
        if clientID != -1:
            print 'Connected to remote API server with port: ', self.port_num, close_all
        else:
            print 'Failed connecting to remote API server with port: ', self.port_num

        self.clientID = clientID
        # vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
        vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
        time.sleep(1)
        # return clientID

    def disconnect_vrep(self):
        vrep.simxFinish(self.clientID)
        print ('Program ended')


    ########################################################################################################################################
    ###################################   interface function to communicate to the simulator ###############################################
    def call_sim_function(self, object_name, function_name, input_floats=[]):
        inputInts = []
        inputFloats = input_floats
        inputStrings = []
        inputBuffer = bytearray()
        res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID, object_name,vrep.sim_scripttype_childscript,
                    function_name, inputInts, inputFloats, inputStrings,inputBuffer, vrep.simx_opmode_blocking)

        # print 'function call: ', self.clientID

        return res, retInts, retFloats, retStrings, retBuffer
        
    def get_laser_points(self):
        res,retInts,retFloats,retStrings,retBuffer = self.call_sim_function('LaserScanner_2D', 'get_laser_points')
        return retFloats

    def get_global_path(self):
        res,retInts, path_raw, retStrings, retBuffer = self.call_sim_function('rwRobot', 'get_global_path')
        path_dist = []
        path_angle = []

        for i in range(2, len(path_raw), 2):       
            path_dist.append(path_raw[i])
            path_angle.append(path_raw[i+1])
        
        return path_dist, path_angle


    def convert_state(self, laser_points, current_pose, path):
        path = np.asarray(path)
        laser_points = np.asarray(laser_points)
        state = np.append(path, laser_points)

        # state = np.asarray(path)
        # state = state.flatten()
        return state

        
    def reset(self):
        self.reached_index = -1
        res,retInts,retFloats,retStrings,retBuffer = self.call_sim_function('rwRobot', 'reset')
        state, reward, is_finish, info = self.step([0,0,0,0,0])
        return state

    def step(self, action):
        if isinstance(action, np.int32) or isinstance(action, int):
            action = action_list[action]

        res, retInts, current_pose, retStrings, found_pose = self.call_sim_function('rwRobot', 'step', action)

        laser_points = self.get_laser_points()
        path_x, path_y = self.get_global_path()  # the target position is located at the end of the list
        

        #compute reward and is_finish
        ########################################################################################################################################
        reward, is_finish  = self.compute_reward(action, path_x, path_y, found_pose)
        ########################################################################################################################################

        path_f = []
        sub_path = [path_x[-1], path_y[-1]] # target x, target y (or angle)
        path_f.append(sub_path)

        state_ = self.convert_state(laser_points, current_pose, path_f)

        return state_, reward, is_finish, ''


    ###################################################  reward function ###################################################################
    def compute_reward(self, action, paty_x, path_y, found_pose):
        is_finish = False
        reward = 0

        dist = math.sqrt(paty_x[-1]*paty_x[-1] + path_y[-1]*path_y[-1])
        # dist = paty_x[-1]
        if dist < self.dist_pre:  # when closer to target
            reward = 1
        else:
            reward = -1

        self.dist_pre = dist

        if dist < 0.1:              # when reach to the target
            is_finish = True
            reward = 5

        if dist > 5:                # when too far away to the target
            is_finish = True
            reward = -1

        if found_pose == 'f':       # when collision or no pose can be found
            # is_finish = True  
            reward = -5

        reward -= 1

        return reward, is_finish