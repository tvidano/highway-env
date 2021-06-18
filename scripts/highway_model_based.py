# Environment
import gym
import sys

# Models and computation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
# torch.set_default_tensor_type("torch.cuda.FloatTensor")

# Visualization
import matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import trange
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from gym.wrappers import Monitor
import base64

# IO
from pathlib import Path
import time

if 'highway_env' not in sys.modules:
    try:
        import highway_env
    except ImportError:
        sys.path.append('/home/trevor/Documents/Python_files/highway-env')
        import highway_env

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    #env = highway_env.envs.merge_env.MergeEnv()
    env = highway_env.envs.collision_env.CollisionEnv()
    #env = highway_env.envs.highway_env.HighwayEnv()
    #env = highway_env.envs.exit_env.ExitEnv()
    #env = highway_env.envs.two_way_env.TwoWayEnv()
    #env = highway_env.envs.lane_keeping_env.LaneKeepingEnv() # can't get this to work with manual
    #env = highway_env.envs.u_turn_env.UTurnEnv()
    #env = gym.make('highway-v0')
    env.configure({
        "manual_control": False,
        "action": {
            "type": "ContinuousAction", # DiscreteMetaAction
            "dynamical": False # there is something really wrong with their dynamics
        },
        "simulation_frequency": 50,  # [Hz]
        "policy_frequency": 10,  # [Hz]
        "offroad_terminal": True,
        "lanes_count": 2,
        "vehicles_density": 1.5,
        "vehicles_count": 30,
        "controlled_vehicles": 1,
        "screen_width": 900,  # [px]
        "screen_height": 150,  # [px]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "stopping_vehicles_count": 2,
        "duration": 20,  # [s]
    })
    env.reset()
    
    for t in range(500):
        env.render()
        #action = env.action_space.sample()
        action = np.array([-1, 0])
        obs, rew, done, info = env.step(action)
        if done:
            print(f'Finished after {t+1} steps.')
            break
    env.close()