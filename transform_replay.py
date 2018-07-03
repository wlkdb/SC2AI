#!/usr/bin/env python

# run "python transform_replay.py --agent * --replay * "

from absl import app
from absl import flags
from pysc2.lib import features, point
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
import importlib
import os.path
import time

FLAGS = flags.FLAGS
flags.DEFINE_string("replay", None, "Path to a replay file or folder.")
flags.DEFINE_string("agent", None, "Path to an agent.")
flags.mark_flag_as_required("replay")
flags.mark_flag_as_required("agent")

class ReplayEnv:
    def __init__(self,
                 agent,
                 discount=1.,
                 step_mul=8):

        self.agent = agent
        self.discount = discount
        self.step_mul = step_mul

        self.run_config = run_configs.get()
        self.sc2_proc = self.run_config.start()
        self.controller = self.sc2_proc.controller

    @staticmethod
    def _valid_replay(info, ping):
        """Make sure the replay isn't corrupt, and is worth looking at."""
        print("replay_version = ", info.base_build)
        print("client_version = ", ping.base_build)
        return True

    def openReplay(self, 
                   replay_file_path,
                   player_id=1,
                   screen_size_px=(84, 84),
                   minimap_size_px=(64, 64)):
        replay_data = self.run_config.replay_data(replay_file_path)
        ping = self.controller.ping()
        info = self.controller.replay_info(replay_data)
        if not self._valid_replay(info, ping):
            raise Exception("{} is not a valid replay file!".format(replay_file_path))

        screen_size_px = point.Point(*screen_size_px)
        minimap_size_px = point.Point(*minimap_size_px)
        interface = sc_pb.InterfaceOptions(
            raw=False, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=24))
        screen_size_px.assign_to(interface.feature_layer.resolution)
        minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)

        map_data = None
        if info.local_map_path:
            map_data = self.run_config.map_data(info.local_map_path)

        self._episode_length = info.game_duration_loops
        self._episode_steps = 0

        self.controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

        self._state = StepType.FIRST
        
    def closeReplay(self):
        return
        
    def start(self, replay_file_path):
        self.openReplay(replay_file_path)
        
        _features = features.Features(self.controller.game_info())

        while True:
            self.controller.step(self.step_mul)
            obs = self.controller.observe()
            agent_obs = _features.transform_obs(obs.observation)

            if obs.player_result: 
                self._state = StepType.LAST
                discount = 0
            else:
                discount = self.discount

            self._episode_steps += self.step_mul

            step = TimeStep(step_type=self._state, reward=0,
                            discount=discount, observation=agent_obs)

            self.agent.step(step, obs.actions, self._state == StepType.LAST)

            if obs.player_result:
                break

            self._state = StepType.MID
        
        self.closeReplay()


def main(unused):
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)

    input_path = FLAGS.replay
    print("replay = ", input_path, "    ", os.path.isdir(input_path))
    
    env = ReplayEnv(agent_cls())
    for loop in range(0, 10):
        if os.path.isdir(input_path):
            list = os.listdir(input_path)
            print("files = ", list)
            for i in range(0, len(list)):   
                path = os.path.join(input_path, list[i])      
                print("new replay = ", path)
                if os.path.isfile(path):             
                    env.start(path)
        else:
            env.start(input_path)

if __name__ == "__main__":
    app.run(main)
