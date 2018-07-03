# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 06:30:32 2018

@author: lg
"""

import tensorflow as tf

slim = tf.contrib.slim

class Model():
        
    time_step = 0
    learning_rate = 0.001
    
    session = None
    
    def createQNetwork_scr(self, value_types, size, deep):
        self._status_input_scr = tf.placeholder("float", [1, size, size, deep])
        net = slim.conv2d(self._status_input_scr, 16, [5, 5], stride=1, padding='SAME', scope="conv_1")
        net = slim.conv2d(net, 16, [5, 5], stride=1, padding='SAME', scope="conv_2")
        net = slim.conv2d(net, 16, [1, 1], stride=1, padding='SAME', scope="conv_3")
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm_scr')
        net = slim.conv2d(net, value_types, 1, stride=1, activation_fn=None, normalizer_fn=None, scope='logits_scr')
        self.QValue_scr = net[0]
        
        self._input_value_scr = tf.placeholder("float")  
        
        labels = self._last_xy[1] * size * value_types + self._last_xy[0] * value_types + self._last_action
        net = tf.reshape(net, [-1]) 
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = net, labels = labels, name = 'cross_entropy_per_example_scr')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy_scr')
    
        cost_input_value = tf.square(self._input_value_scr - self.QValue_scr[self._last_xy[1]][self._last_xy[0]][self._last_action])
        
        self.cost_scr = tf.cond(self._use_softmax, lambda: cross_entropy_mean, lambda: cost_input_value)  
    
    def createQNetwork(self, value_types, CHECKFILE_DIR):
        
        self._last_action = tf.placeholder("int32")
        self._last_xy = tf.placeholder("int32", [2])
        self._use_softmax = tf.placeholder("bool")  
        self._learning_rate = tf.placeholder("float")     
            
        self.createQNetwork_scr(value_types, 84, 7)
        self.trainStep = tf.train.AdamOptimizer(self._learning_rate).minimize(self.cost_scr)
        
        # saving and loading networks
        saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(CHECKFILE_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.session, checkpoint.model_checkpoint_path)
            self.time_step = int(checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print("Successfully loaded: ", checkpoint.model_checkpoint_path)
            print("now step: ", self.time_step)
        else:
            print("Could not find old network weights")
            
            
            

            
            
            
            