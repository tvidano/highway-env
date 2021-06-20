# Environment
import gym
import sys

# Models and computation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
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
            "dynamical": False
        },
        "simulation_frequency": 40,  # [Hz]
        "policy_frequency": 10,  # [Hz]
        "offroad_terminal": True,
        "lanes_count": 3,
        "vehicles_density": 2,
        "vehicles_count": 15,
        "controlled_vehicles": 1,
        "screen_width": 900,  # [px]
        "screen_height": 150,  # [px]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "stopping_vehicles_count": 5,
        "duration": 10,  # [s]
    })

    #check_env(env)

    #model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=2048,
    #            batch_size=64, n_epochs=10,verbose=1)
    #model.learn(total_timesteps=25000, )
    #model.save("ppo_collision")
    #model.load("ppo_collision")

    obs = env.reset()
    model_params = []
    rewards = []
    done = False
    while not done:
        #action = env.action_space.sample()
        action = np.array([-1, 0])
        #action, _states = model.predict(obs)
        obs, rew, done, info = env.step(action)
        rewards.append(rew)
        #if done:
        #    print(f'Finished after {t+1} steps.')
        #    break
        env.render()
    env.close()
    
    plt.plot(rewards)
    plt.show()
