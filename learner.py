# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 05:39:31 2018

@author: lg
"""

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import tensorflow as tf
import random
import time
import numpy as np
import model
import info

# Functions
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id 
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id

_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id 

_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_RALLY_UNITS_SCREEN = actions.FUNCTIONS.Rally_Units_screen.id


_MAP_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_MAP_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index

# Unit IDs
_TERRAN_BASE = 18
_TERRAN_SUPPLYDEPOT = 19
_TERRAN_BARRACKS = 21

_TERRAN_SCV = 45

_CRYSTAL = 483

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]
_SINGLE = [0]
_ALL_TYPE = [2]
_RECALL = [0]
_SET = [1] 

_SUPPLY_USERD = 3
_SUPPLY_MAX = 4
_SUPPLY_WORKERS = 6

# Strings
_SCR = "screen"
_PLY = "player"
_MAP = "minimap"
_AVA_ACT = "available_actions"
_SELECT = "single_select"
_SELECTS = "multi_select"
_TRAINS = "build_queue"
_GROUPS = "control_groups"

CHECKFILE_DIR='./checkpoint/'
CHECKFILE = CHECKFILE_DIR + 'model.ckpt'

class LG(base_agent.BaseAgent):
    
    is_test = True
    
    save_when_game_end = True
    
    size_scr = 84
    size_map = 64
    x_space = 0 
    y_space = 0 
    
    # DQN model
    model = new_model.Model()
    time_step = 0
    FINAL_EPSILON = 0.01 # final value of epsilon
    INITIAL_EPSILON = 0.4  # starting value of epsilon
    EPSILON_CHANGE = 0.95
    GAMMA = 0.9 # decay rate of past observations
    value_types = 2
    reward_plus = 0
    
    # status
    status = info.StepStatus(size_scr, size_map)
    last_status = None
    
    is_last = False
    selected_stay_time = np.zeros([size_scr, size_scr], dtype = np.int32)
    map_hidden_time = np.zeros([size_map, size_map], dtype = np.int32)

    def updateStatus(self):
        self.status.reset()
        self.status.update(self.obs)
        
    def getMaxQAndActionsIn(self, QValue, size, ava_actions):
        max_Q = None
        actions = []
        for y in range(self.y_space, size - self.y_space):
            for x in range(self.x_space, size - self.x_space):
                for a in ava_actions:
                    if max_Q == None or QValue[y][x][a] > max_Q:
                        max_Q = QValue[y][x][a]
                        actions = [[[x, y], a]]
                    elif QValue[y][x][a] == max_Q:
                        actions.append([[x, y], a])  
        
        if (self.model.time_step + 1) % 50 == 0 and size == 84:
            for y in range(0, size, 4):
                for x in range(0, size, 4):
					#hit_points_selected
                    print(int(self.status.input_scr[y][x][2]), end = " ")
                print()
                    
            for y in range(0, size, 4):
                for x in range(0, size, 4):
                    print(int(QValue[y][x][0] * 10)/10, end = " ")
                print()
            
        return max_Q, actions
    
    def randomChoiceIn(self, actions, size):
        if not self.is_test and self.epsilon >= random.random():
            x = random.randint(self.x_space, size - 1 - self.x_space)
            y = random.randint(self.y_space, size - 1 - self.y_space)
            a = random.randint(0, self.value_types - 1)
            return [[x, y], a]
        return random.choice(actions)
    
    def updateLearningRate(self):
        self.learning_rate = 0.001 
        
    def getAction(self, actions = None):
        start_time = time.time()
        
        _QValue_scr = self.model.QValue_scr.eval(feed_dict={self.model._status_input_scr: [self.status.input_scr]})
        ava_actions = range(self.value_types)
        self.status.max_Q_scr, _actions_scr = self.getMaxQAndActionsIn(_QValue_scr, self.size_scr, ava_actions)  
                       
        self.status.xya = None
        if actions is None:
            self.status.xya = self.randomChoiceIn(_actions_scr, self.size_scr)
        else:
            self.is_test = False
            action_type = -1
            if len(actions) > 0 and actions[0].action_feature_layer != None:
#                print("actions = ", actions)
                action = actions[0].action_feature_layer.unit_command
                if action.ability_id == 23:
                    action_type = 0
                elif action.ability_id == 1:
                    action_type = 1
            if action_type >= 0:
                x = min(self.size_scr - self.x_space, max(self.x_space, action.target_screen_coord.x))
                y = min(self.size_scr - self.y_space, max(self.y_space, action.target_screen_coord.y))
                self.status.xya = [[x, y], action_type]                 
                
        duration = time.time() - start_time
        if self.status.xya != None and (self.model.time_step + 1) % 10 == 0: 
            self.status.Q_scr = _QValue_scr[self.status.xya[0][1]][self.status.xya[0][0]][self.status.xya[1]]
            print("step ", self.model.time_step)
            print("move to ", self.status.xya,"      Q_scr: ", self.status.Q_scr, "    maxQ_scr: ", self.status.max_Q_scr, "   time: ", duration)

    def updateLastQValue(self):
        self.model.time_step += 1
        if not self.is_test and (self.model.time_step % 1000 == 0 or (self.save_when_game_end and self.is_last)):
            saver = tf.train.Saver()
            saver.save(self.model.session, CHECKFILE, global_step = self.model.time_step)
            
        if self.last_status == None or self.last_status.xya == None or self.is_test:
            return
        
        start_time = time.time()
        reward = (self.status.score - self.last_status.score) * 1000 + self.reward_plus 
        input_value_scr = reward + self.GAMMA * self.status.max_Q_scr
        
        if self.last_status.xya != None:
            self.updateLearningRate()
            self.model.trainStep.run(feed_dict={
                self.model._learning_rate: self.learning_rate,
                self.model._last_xy: self.last_status.xya[0],
                self.model._last_action: self.last_status.xya[1],
                self.model._use_softmax: True,
                self.model._status_input_scr: [self.last_status.input_scr],
                self.model._input_value_scr: input_value_scr
                })
        duration = time.time() - start_time
        
    def updateLaststatus(self):
        self.last_status = self.status
        
    def isAvailable(self, action):
        return action in self.obs.observation[_AVA_ACT]
    
    def step(self, obs, action = None, is_last = False):
        super(LG, self).step(obs)
        self.obs = obs
        self.is_last = is_last
        if action is None and not self.isAvailable(_MOVE_SCREEN):
            if self.isAvailable(_SELECT_ARMY):
                print("select army")
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            else:
                return actions.FunctionCall(_NOOP, [])
        
        self.updateStatus()
        self.getAction(action)
        self.updateLastQValue()
        self.updateLaststatus()
        
        if action is None:
#           return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, self.status.xya[0]])
            return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, self.status.xya[0]])
    
    def __init__(self):
        super(LG, self).__init__()
        # init some parameters
        self.epsilon = self.INITIAL_EPSILON
        self.learning_rate = 0.001 
        # init Q network
        self.model.createQNetwork(self.value_types, CHECKFILE_DIR)

    def reset(self):
        super(LG, self).reset()
        self.epsilon *= self.EPSILON_CHANGE
        if self.epsilon < self.FINAL_EPSILON:
            self.epsilon = self.FINAL_EPSILON
        print("epsilon = ", self.epsilon)
        print("learning rate = ", self.model.learning_rate)
        
        self.last_status = None
        
        
    