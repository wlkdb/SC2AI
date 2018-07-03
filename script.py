# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 05:39:31 2018

@author: lg
"""

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time
import random
import Queue
import numpy

# Functions
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id 
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id

_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id 

_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_RALLY_UNITS_SCREEN = actions.FUNCTIONS.Rally_Units_screen.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

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

class LG(base_agent.BaseAgent):
    base_top_left = None
    
    scv_selected = False
    barracks_selected = False
    
    base_rallied = False
    barracks_rallied = False
    
    supply_num = 0
    barracks_num = 0
    
    build_supply_count = 0
    build_barrack_count = 0
    train_scv_count = 0
    train_marine_count = 0
    
    queue_actions = Queue.Queue()
    queue_params = Queue.Queue()
    
    f = open("out.txt", "w")    
    
    def getLocation(self, unit):
        unit_type = self.obs.observation[_SCR][_UNIT_TYPE]
        return (unit_type == unit).nonzero()    
    
    def transformLocation(self, x, x_bias, y, y_bias):
        if not self.base_top_left:
            x -= x_bias
            y -= y_bias
        else:
            x += x_bias
            y += y_bias
        x = min(max(0, x), self.screen_width - 1)
        y = min(max(0, y), self.screen_height - 1)
        return [x, y]
    
    def selectUnit(self, unit):
        unit_y, unit_x = self.getLocation(unit)
        if not unit_y.any():
            return actions.FunctionCall(_NOOP, [])
        i = random.randint(0, len(unit_y) - 1)
        target = [unit_x[i], unit_y[i]]       
        return actions.FunctionCall(_SELECT_POINT, [_SINGLE, target])
    
    def selectBuilding(self, unit):
        unit_y, unit_x = self.getLocation(unit)
        if not unit_y.any():
            return None
        i = random.randint(0, len(unit_y) - 1)
        target = [unit_x[i], unit_y[i]]            
        return actions.FunctionCall(_SELECT_POINT, [_ALL_TYPE, target])
    
    def scvBackToGather(self):
        unit_y, unit_x = self.getLocation(_TERRAN_BASE)
        target = self.transformLocation(int(unit_x.mean()), -18, int(unit_y.mean()), -18)
        self.queue_actions.put(_GATHER)
        self.queue_params.put([_QUEUED, target])
        
    def isAvailable(self, action):
        return action in self.obs.observation[_AVA_ACT]
        
    def step(self, obs):
        super(LG, self).step(obs)
        self.obs = obs
        
        time.sleep(0.5)
        self.build_supply_count = max(0, self.build_supply_count - 1)
        self.build_barrack_count = max(0, self.build_barrack_count - 1)
        self.train_marine_count = max(0, self.train_marine_count - 1)
        self.train_scv_count = max(0, self.train_scv_count - 1)

        numpy.set_printoptions(threshold='nan')
        
        self.f = open("out.txt", "w")    
        print >> self.f, ""
        print >> self.f, "SCR UNIT_TYPE = "
        for row in obs.observation[_SCR][_UNIT_TYPE]:
            print >> self.f, row
        print  >> self.f, ""
        self.f.close()
        
        # get base orientation
        if self.base_top_left is None:
            player_y, player_x = (obs.observation[_MAP][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31
            self.screen_width = len(obs.observation[_SCR][0][0])
            self.screen_height = len(obs.observation[_SCR][0])
                                          
        while not self.queue_actions.empty():
            action = self.queue_actions.get()
            param = self.queue_params.get()
            if self.isAvailable(action):
                return actions.FunctionCall(action, param)
        
        # train SCV
        if obs.observation[_PLY][_SUPPLY_USERD] < obs.observation[_PLY][_SUPPLY_MAX] and\
                          self.train_scv_count == 0 and obs.observation[_PLY][_SUPPLY_WORKERS] < 19:
            if obs.observation[_SELECT][0][0] != _TERRAN_BASE:
                action = self.selectBuilding(_TERRAN_BASE)
                if action != None:
                    return action
            elif _TRAIN_SCV in obs.observation[_AVA_ACT] and len(obs.observation[_TRAINS]) <= 1:
                self.train_scv_count += 20
                return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])
        
        # build supply
        if (self.supply_num == 0 or obs.observation[_PLY][_SUPPLY_USERD] + 1 + self.barracks_num >= obs.observation[_PLY][_SUPPLY_MAX])\
           and self.build_supply_count == 0:
            if _BUILD_SUPPLYDEPOT in obs.observation[_AVA_ACT]:
                unit_y, unit_x = self.getLocation(_TERRAN_BASE)
                target = self.transformLocation(int(unit_x.mean()), self.supply_num % 3 * 7, int(unit_y.mean()), 12 + self.supply_num / 3 * 7)
                self.supply_num += 1
                self.build_supply_count += 80
                self.scvBackToGather()
                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])
            else:
                return self.selectUnit(_TERRAN_SCV)
        
        # build barracks
        if self.barracks_num < 6 and self.build_barrack_count == 0:
            if _BUILD_BARRACKS in obs.observation[_AVA_ACT]:
                unit_y, unit_x = self.getLocation(_TERRAN_BASE)
                target = self.transformLocation(int(unit_x.mean()), 15 + self.barracks_num % 3 * 12, int(unit_y.mean()), -(self.barracks_num / 3 * 12)) 
                self.barracks_num += 1
                self.scvBackToGather()
                self.build_barrack_count += 15
                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
            else:
                return self.selectUnit(_TERRAN_SCV)
        
        # set barracks rally
        if not self.barracks_rallied:
            if obs.observation[_SELECT][0][0] != _TERRAN_BARRACKS:
                action = self.selectBuilding(_TERRAN_BARRACKS)
                if action != None:
                    return action
            else:
                self.barracks_rallied = True
                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, 
                                            [_NOT_QUEUED, [29, 21 if self.base_top_left else 46]])
        
        # train marine
        if obs.observation[_PLY][_SUPPLY_USERD] < obs.observation[_PLY][_SUPPLY_MAX] and self.train_marine_count == 0:
            if not _TRAIN_MARINE in obs.observation[_AVA_ACT]: 
                action = self.selectBuilding(_TERRAN_BARRACKS) 
                if action == None:
                    if self.isAvailable(_SELECT_CONTROL_GROUP):
                        action = actions.FunctionCall(_SELECT_CONTROL_GROUP, [_RECALL, [3]])
                    else:
                        return actions.FunctionCall(_NOOP, [])
                group_type, group_count = obs.observation[_GROUPS][3]
                if group_type != _TERRAN_BARRACKS or group_count < self.barracks_num:
                    self.queue_actions.put(_SELECT_CONTROL_GROUP)
                    self.queue_params.put([_SET, [3]])
                self.queue_actions.put(_RALLY_UNITS_MINIMAP)
                self.queue_params.put([_NOT_QUEUED, [29, 21 if self.base_top_left else 46]])
                return action
            else: 
                self.train_marine_count += 6
                self.queue_actions.put(_TRAIN_MARINE)
                self.queue_params.put([_QUEUED])
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
            
        if len(obs.observation[_SELECTS]) < 24:
            if _SELECT_ARMY in obs.observation["available_actions"]:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
            if self.base_top_left:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])
            else:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])
        
        return actions.FunctionCall(_NOOP, [])