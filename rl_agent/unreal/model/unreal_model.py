# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import sys


# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_initializer(input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels)
        return tf.random_uniform(shape, minval=-d, maxval=d)
    return _initializer


def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
        return tf.random_uniform(shape, minval=-d, maxval=d)
    return _initializer


class UnrealModel(object):
    """
    UNREAL algorithm network model.
    """
    def __init__(self,
                action_size,
                thread_index, # -1 for global
                use_pixel_change,
                use_value_replay,
                use_reward_prediction,
                pixel_change_lambda,
                entropy_beta,
                device,
                for_display=False):
        self._device = device
        self._action_size = action_size
        self._thread_index = thread_index
        self._use_pixel_change = use_pixel_change
        self._use_value_replay = use_value_replay
        self._use_reward_prediction = use_reward_prediction
        self._pixel_change_lambda = pixel_change_lambda
        self._entropy_beta = entropy_beta
        
        self._create_network(for_display)
    
    def _create_network(self, for_display):
        scope_name = "net_{0}".format(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            # # lstm
            # self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            
            # [base A3C network]
            self._create_base_network()

            # # [Pixel change network]
            # if self._use_pixel_change:
            #     self._create_pc_network()
            #     if for_display:
            #         self._create_pc_network_for_display()

            # # [Value replay network]
            # if self._use_value_replay:
            #     self._create_vr_network()

            # # [Reawrd prediction network]
            # if self._use_reward_prediction:
            #     self._create_rp_network()
            
            # self.reset_state()

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)


    def _create_base_network(self):
        # State (Base image input)
        self.base_input = tf.placeholder("float", [None, 182])

        # Last action and reward
        self.base_last_action_reward_input = tf.placeholder("float", [None, self._action_size+1])
        
        # feature extration layers
        # self.base_path_feature_output = self._base_fc_layers(self.
        self.base_feature_output = self._base_feature_layers(self.base_input)

        self.base_pi = self._base_policy_layer(self.base_feature_output) # policy output
        self.base_v  = self._base_value_layer(self.base_feature_output)  # value output

    def _base_feature_layers(self, state_input, reuse=False):
        with tf.variable_scope("base_conv", reuse=reuse) as scope:
            self.path = tf.slice(state_input, [0, 0], [1, 2])
            self.laser = tf.slice(state_input, [0, 2], [1, 180])
            laser_reshape = tf.reshape(self.laser, [-1, 180, 1])
            conv1 = tf.layers.conv1d(   inputs=laser_reshape,
                                        filters=16,
                                        kernel_size=8,
                                        padding="same",
                                        activation=tf.nn.relu,
                                        name="conv1")

            conv2 = tf.layers.conv1d(   inputs=conv1,
                                        filters=32,
                                        kernel_size=4,
                                        padding="same",
                                        activation=tf.nn.relu,
                                        name="conv2")
            conv_flat = tf.reshape(conv2, [-1, 1 * 180 * 32])
            path_flat = tf.reshape(self.path, [-1, 2])
            fc1 = tf.layers.dense(inputs=path_flat, units=64, activation=tf.nn.relu)

            feature = tf.concat([conv_flat, fc1], 1)
            return (feature)

    def run_path_laser(self, sess, s_t):
        # This run_base_policy_and_value() is used when forward propagating.
        # so the step size is 1.
        # sess.run( [self.path_laser], feed_dict = {self.base_input : [s_t]})
        # pi_out: (1,3), v_out: (1)
        print(sess.run( [self.path_laser], feed_dict = {self.base_input : [s_t]}))
        result = sess.run( [self.path_laser], feed_dict = {self.base_input : [s_t]})
        # print ((result[0][0].shape), (result[0][1].shape))
        # return path, laser

    def _base_fc_layers(self, state_input, reuse=False):
        with tf.variable_scope("base_fc", reuse=reuse) as scope:
            # Weights
            size = 1 * 180 * 32 + 64
            W_fc1, b_fc1 = self._fc_variable([size, 128], "base_fc1")
            # W_fc2, b_fc2 = self._fc_variable([64, 64], "base_fc2")

            # Nodes
            state_flat = tf.reshape(state_input, [-1, size])

            h_fc1 = tf.nn.relu(tf.matmul(state_flat, W_fc1) + b_fc1)
            # h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
            return h_fc1


    def _base_policy_layer(self, feature_outputs, reuse=False):
        with tf.variable_scope("base_policy", reuse=reuse) as scope:
            # Weight for policy output layer
            size = 1 * 180 * 32 + 64
            W_fc_p, b_fc_p = self._fc_variable([size, self._action_size], "base_fc_p")
            # Policy (output)

            base_pi = tf.nn.softmax(tf.matmul(feature_outputs, W_fc_p) + b_fc_p)
            return base_pi


    def _base_value_layer(self, feature_outputs, reuse=False):
        with tf.variable_scope("base_value", reuse=reuse) as scope:
            # Weight for value output layer
            size = 1 * 180 * 32 + 64
            W_fc_v, b_fc_v = self._fc_variable([size, 1], "base_fc_v")
            
            # Value (output)
            v_ = tf.matmul(feature_outputs, W_fc_v) + b_fc_v
            base_v = tf.reshape( v_, [-1] )
            return base_v


    # def _create_pc_network(self):
    #     # State (Image input) 
    #     self.pc_input = tf.placeholder("float", [None, 84, 84, 3])

    #     # Last action and reward
    #     self.pc_last_action_reward_input = tf.placeholder("float", [None, self._action_size+1])

    #     # pc conv layers
    #     pc_conv_output = self._base_conv_layers(self.pc_input, reuse=True)

    #     # pc lastm layers
    #     pc_initial_lstm_state = self.lstm_cell.zero_state(1, tf.float32)
    #     # (Initial state is always resetted.)
        
    #     pc_lstm_outputs, _ = self._base_lstm_layer(pc_conv_output,
    #                                             self.pc_last_action_reward_input,
    #                                             pc_initial_lstm_state,
    #                                             reuse=True)
        
    #     self.pc_q, self.pc_q_max = self._pc_deconv_layers(pc_lstm_outputs)


    # def _create_vr_network(self):
    #     # State (Image input)
    #     self.vr_input = tf.placeholder("float", [None, 84, 84, 3])

    #     # Last action and reward
    #     self.vr_last_action_reward_input = tf.placeholder("float", [None, self._action_size+1])

    #     # VR conv layers
    #     vr_conv_output = self._base_conv_layers(self.vr_input, reuse=True)

    #     # pc lastm layers
    #     vr_initial_lstm_state = self.lstm_cell.zero_state(1, tf.float32)
    #     # (Initial state is always resetted.)
        
    #     vr_lstm_outputs, _ = self._base_lstm_layer(vr_conv_output,
    #                                             self.vr_last_action_reward_input,
    #                                             vr_initial_lstm_state,
    #                                             reuse=True)
    #     # value output
    #     self.vr_v  = self._base_value_layer(vr_lstm_outputs, reuse=True)

    
    # def _create_rp_network(self):
    #     self.rp_input = tf.placeholder("float", [3, 84, 84, 3])

    #     # RP conv layers
    #     rp_conv_output = self._base_conv_layers(self.rp_input, reuse=True)
    #     rp_conv_output_rehaped = tf.reshape(rp_conv_output, [1,9*9*32*3])
        
    #     with tf.variable_scope("rp_fc") as scope:
    #         # Weights
    #         W_fc1, b_fc1 = self._fc_variable([9*9*32*3, 3], "rp_fc1")

    #         # Reawrd prediction class output. (zero, positive, negative)
    #         self.rp_c = tf.nn.softmax(tf.matmul(rp_conv_output_rehaped, W_fc1) + b_fc1)
    #         # (1,3)

    def _base_loss(self):
        # [base A3C]
        # Taken action (input for policy)
        self.base_a = tf.placeholder("float", [None, self._action_size])
        
        # Advantage (R-V) (input for policy)
        self.base_adv = tf.placeholder("float", [None])
        
        # Avoid NaN with clipping when value in pi becomes zero
        log_pi = tf.log(tf.clip_by_value(self.base_pi, 1e-20, 1.0))
        
        # Policy entropy
        entropy = -tf.reduce_sum(self.base_pi * log_pi, reduction_indices=1)
        
        # Policy loss (output)
        policy_loss = -tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.base_a ),
                                                    reduction_indices=1 ) *
                                    self.base_adv + entropy * self._entropy_beta)
        
        # R (input for value target)
        self.base_r = tf.placeholder("float", [None])
        
        # Value loss (output)
        # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
        value_loss = 0.5 * tf.nn.l2_loss(self.base_r - self.base_v)
        
        base_loss = policy_loss + value_loss
        return base_loss

  
    # def _pc_loss(self):
    #     # [pixel change]
    #     self.pc_a = tf.placeholder("float", [None, self._action_size])
    #     pc_a_reshaped = tf.reshape(self.pc_a, [-1, 1, 1, self._action_size])

    #     # Extract Q for taken action
    #     pc_qa_ = tf.multiply(self.pc_q, pc_a_reshaped)
    #     pc_qa = tf.reduce_sum(pc_qa_, reduction_indices=3, keep_dims=False)
    #     # (-1, 20, 20)
        
    #     # TD target for Q
    #     self.pc_r = tf.placeholder("float", [None, 20, 20])

    #     pc_loss = self._pixel_change_lambda * tf.nn.l2_loss(self.pc_r - pc_qa)
    #     return pc_loss

    
    # def _vr_loss(self):
    #     # R (input for value)
    #     self.vr_r = tf.placeholder("float", [None])
        
    #     # Value loss (output)
    #     vr_loss = tf.nn.l2_loss(self.vr_r - self.vr_v)
    #     return vr_loss


    # def _rp_loss(self):
    #     # reward prediction target. one hot vector
    #     self.rp_c_target = tf.placeholder("float", [1,3])
        
    #     # Reward prediction loss (output)
    #     rp_c = tf.clip_by_value(self.rp_c, 1e-20, 1.0)
    #     rp_loss = -tf.reduce_sum(self.rp_c_target * tf.log(rp_c))
    #     return rp_loss
        
        
    def prepare_loss(self):
        with tf.device(self._device):
            loss = self._base_loss()
            
            if self._use_pixel_change:
                pc_loss = self._pc_loss()
                loss = loss + pc_loss

            if self._use_value_replay:
                vr_loss = self._vr_loss()
                loss = loss + vr_loss

            if self._use_reward_prediction:
                rp_loss = self._rp_loss()
                loss = loss + rp_loss
            
            self.total_loss = loss


    def run_base_policy_and_value(self, sess, s_t, last_action_reward):
        # This run_base_policy_and_value() is used when forward propagating.
        # so the step size is 1.
        pi_out, v_out = sess.run( [self.base_pi, self.base_v], feed_dict = {self.base_input : [s_t]})
        # pi_out: (1,3), v_out: (1)
        return (pi_out[0], v_out[0])

  
    def run_base_policy_value_pc_q(self, sess, s_t, last_action_reward):
        # For display tool.
        pi_out, v_out, self.base_lstm_state_out, q_disp_out, q_max_disp_out = \
            sess.run( [self.base_pi, self.base_v, self.base_lstm_state, self.pc_q_disp, self.pc_q_max_disp],
                    feed_dict = {self.base_input : [s_t], self.base_last_action_reward_input : [last_action_reward]} )
        
        # pi_out: (1,3), v_out: (1), q_disp_out(1,20,20, action_size)
        return (pi_out[0], v_out[0], q_disp_out[0])

  
    def run_base_value(self, sess, s_t, last_action_reward):
        # This run_bae_value() is used for calculating V for bootstrapping at the 
        # end of LOCAL_T_MAX time step sequence.
        # When next sequcen starts, V will be calculated again with the same state using updated network weights,
        # so we don't update LSTM state here.
        v_out = sess.run( [self.base_v], feed_dict = {self.base_input : [s_t]} )
        return v_out[0][0]

    
    def run_pc_q_max(self, sess, s_t, last_action_reward):
        q_max_out = sess.run( self.pc_q_max,
                            feed_dict = {self.pc_input : [s_t],
                                        self.pc_last_action_reward_input : [last_action_reward]} )
        return q_max_out[0]

    
    def run_vr_value(self, sess, s_t, last_action_reward):
        vr_v_out = sess.run( self.vr_v,
                            feed_dict = {self.vr_input : [s_t],
                                        self.vr_last_action_reward_input : [last_action_reward]} )
        return vr_v_out[0]

  
    def run_rp_c(self, sess, s_t):
        # For display tool
        rp_c_out = sess.run( self.rp_c,
                            feed_dict = {self.rp_input : s_t} )
        return rp_c_out[0]

    
    def get_vars(self):
        return self.variables
    

    def sync_from(self, src_netowrk, name=None):
        src_vars = src_netowrk.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name, "UnrealModel",[]) as name:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)
      

    def _fc_variable(self, weight_shape, name):
        name_w = "W_{0}".format(name)
        name_b = "b_{0}".format(name)
        
        input_channels  = weight_shape[0]
        output_channels = weight_shape[1]
        bias_shape = [output_channels]

        weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
        bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
        return weight, bias

  
    def _conv_variable(self, weight_shape, name, deconv=False):
        name_w = "W_{0}".format(name)
        name_b = "b_{0}".format(name)
        
        w = weight_shape[0]
        h = weight_shape[1]
        if deconv:
            input_channels  = weight_shape[3]
            output_channels = weight_shape[2]
        else:
            input_channels  = weight_shape[2]
            output_channels = weight_shape[3]
        bias_shape = [output_channels]

        weight = tf.get_variable(name_w, weight_shape,
                                initializer=conv_initializer(w, h, input_channels))
        bias   = tf.get_variable(name_b, bias_shape,
                                initializer=conv_initializer(w, h, input_channels))
        return weight, bias

  
    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")


    def _get2d_deconv_output_size(self,
                                    input_height, input_width,
                                    filter_height, filter_width,
                                    stride, padding_type):
        if padding_type == 'VALID':
            out_height = (input_height - 1) * stride + filter_height
            out_width  = (input_width  - 1) * stride + filter_width
        
        elif padding_type == 'SAME':
            out_height = input_height * row_stride
            out_width  = input_width  * col_stride
        
        return out_height, out_width


    def _deconv2d(self, x, W, input_width, input_height, stride):
        filter_height = W.get_shape()[0].value
        filter_width  = W.get_shape()[1].value
        out_channel   = W.get_shape()[2].value
        
        out_height, out_width = self._get2d_deconv_output_size(input_height,
                                                            input_width,
                                                            filter_height,
                                                            filter_width,
                                                            stride,
                                                            'VALID')
        batch_size = tf.shape(x)[0]
        output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
        return tf.nn.conv2d_transpose(x, W, output_shape,
                                    strides=[1, stride, stride, 1],
                                    padding='VALID')