# -*- coding: utf-8 -*-
import numpy as np
import math
import tensorflow as tf
from unreal_model import UnrealModel

class TestUnrealModel(tf.test.TestCase):
  
  # feature: conv = 3*2, fc = 2*2, pi: fc = 1*2 v: fc = 1*2
  def test_base_unreal_variable_size(self):
    """ Check base variable size with all options OFF """
    self.check_model_var_size( False,
                               False,
                               False,
                               14 )

  def test_unreal_variable_size(self):
    """ Check total variable size with all options ON """
    use_pixel_change = True
    use_value_replay = True
    use_reward_prediction = True

    self.check_model_var_size( use_pixel_change,
                               use_value_replay,
                               use_reward_prediction,
                               16 )

  def test_vr_variable_size(self):
    """ Check total variable size with only value funciton replay ON """
    use_pixel_change = False
    use_value_replay = True
    use_reward_prediction = False

    # feature: conv = 3*2, fc = 2*2, pi: fc = 1*2 v: fc = 1*2
    self.check_model_var_size( use_pixel_change,
                               use_value_replay,
                               use_reward_prediction,
                               14 )


  def test_rp_variable_size(self):
    """ Check total variable size with only reward prediction ON """
    use_pixel_change = False
    use_value_replay = False
    use_reward_prediction = True

    # feature: conv = 3*2, fc = 2*2, pi: fc = 1*2 v: fc = 1*2
    # rp:   fc=2
    self.check_model_var_size( use_pixel_change,
                               use_value_replay,
                               use_reward_prediction,
                               16 )
    
  def check_model_var_size(self,
                           use_pixel_change,
                           use_value_replay,
                           use_reward_prediction,
                           var_size):
    """ Check variable size of the model """
    
    model = UnrealModel(1,
                        -1,
                        use_pixel_change,
                        use_value_replay,
                        use_reward_prediction,
                        1.0,
                        1.0,
                        "/cpu:0");
    variables = model.get_vars()
    for i in variables:
          print i.name
    print '  '
    self.assertEqual( len(variables), var_size )


if __name__ == "__main__":
  tf.test.main()
  
