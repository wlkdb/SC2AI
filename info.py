# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 07:05:09 2017

@author: lg
"""

from pysc2.lib import features
import numpy as np

_SCR = "screen"

# Features
_SCR_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SCR_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SCR_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_SCR_UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index
_SCR_SELECTED = features.SCREEN_FEATURES.selected.index

class StepStatus:
        

    def __init__(self, size_scr, size_map):
        super(StepStatus, self).__init__()
        self.reset()
        self.size_scr = size_scr
        self.size_map = size_map
        
    def reset(self):
        self.xya = None
        self.score = None
        self.input_act = None
        self.input_scr = None
        self.input_map = None
        self.max_Q_act = None
        self.max_Q_scr = None
        self.max_Q_map = None
        self.Q_act = None
        self.Q_scr = None
        self.Q_map = None
        
    def update(self, obs):
        unit_type = obs.observation[_SCR][_SCR_UNIT_TYPE]
        relative = obs.observation[_SCR][_SCR_PLAYER_RELATIVE]
        hit_points = obs.observation[_SCR][_SCR_UNIT_HIT_POINTS]
        selected = obs.observation[_SCR][_SCR_SELECTED]
        
        unit_type_selected = np.zeros([self.size_scr, self.size_scr])
        unit_type_unselected = np.zeros([self.size_scr, self.size_scr])
        relative_selected = np.zeros([self.size_scr, self.size_scr])
        relative_unselected = np.zeros([self.size_scr, self.size_scr])
        hit_points_selected = np.zeros([self.size_scr, self.size_scr])
        hit_points_unselected = np.zeros([self.size_scr, self.size_scr])
        
        selected_x = []
        selected_y = []
        
        for y in range(self.size_scr):
            for x in range(self.size_scr):
                if selected[y][x] > 0:
                    unit_type_selected[y][x] = unit_type[y][x]
                    relative_selected[y][x] = relative[y][x]
                    hit_points_selected[y][x] = hit_points[y][x]
                    
                    selected_x.append(x)
                    selected_y.append(y)
                else:
                    unit_type_unselected[y][x] = unit_type[y][x]
                    relative_unselected[y][x] = relative[y][x]
                    hit_points_unselected[y][x] = hit_points[y][x]
        if np.size(selected_x) > 0:
            mean_selected_x = int(np.mean(selected_x))
            mean_selected_y = int(np.mean(selected_y))
        else:
            mean_selected_x = 0
            mean_selected_y = 0
        
        vec_mean_selected = np.array([mean_selected_x, mean_selected_y])
        
        dis_to_selected = np.zeros([self.size_scr, self.size_scr])
        for y in range(self.size_scr):
            for x in range(self.size_scr):
                vec_xy = np.array([x, y])
                dis_to_selected[y][x] = int(np.linalg.norm(vec_xy - vec_mean_selected))
#                print(dis_to_selected[y][x], end = " ")
#            print("")
        
        self.input_scr = []
        self.input_scr.append(unit_type_selected)
        self.input_scr.append(relative_selected)
        self.input_scr.append(hit_points_selected)
        self.input_scr.append(unit_type_unselected)
        self.input_scr.append(relative_unselected)
        self.input_scr.append(hit_points_unselected)
        self.input_scr.append(dis_to_selected)
        
        self.input_scr = np.transpose(self.input_scr, (1, 2, 0))
        self.score = obs.observation['score_cumulative'][0]
                