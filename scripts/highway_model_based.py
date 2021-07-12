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
    env = highway_env.envs.collision_env.CollisionEnv()

    # Override standard configuration
    # env.configure({
    #     "manual_control": False,
    #     "simulation_frequency": 15,  # [Hz]
    #     "policy_frequency": 10,  # [Hz]
    #     "vehicles_density": 2,
    #     "vehicles_count": 20,
    #     "stopping_vehicles_count": 2,
    #     "duration": 10,  # [s]
    # })

    # Uncomment to check environment with OpenAi Gym:
    # check_env(env)

    # Uncomment to try training an PPO algorithm on this environemt:
    # model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=2048,
    #            batch_size=64, n_epochs=10,verbose=1)
    # model.learn(total_timesteps=25000, )
    # model.save("ppo_collision")
    # model.load("ppo_collision")

    obs = env.reset()
    env.render()
    model_params = []
    rewards = []
    velocity = []
    forces = []
    slips = []
    ttc = []
    done = False
    #TODO{tvidano}: Figure out why vehicles no longer stop
    while not done:
        # Use just hard braking (will probably lock the wheel):
        action = np.array([-1, 0.0])

        # Uncomment to use RL algorithm actions:
        #action, _states = model.predict(obs)

        obs, rew, done, info = env.step(action)
        # velocity.append(info["speed"])
        # forces.append(info["tire_forces"])
        # slips.append(info["slip_values"])
        # ttc.append(info["ttc"])
        rewards.append(rew)
        
    env.close()
    # velocity = np.vstack(velocity)
    # forces = np.vstack(forces)
    # slips = np.vstack(slips)
    
    # plt.figure()
    # plt.subplot(231)
    # plt.plot(velocity[:,0])
    # plt.title('Long. Vel.')
    # plt.subplot(232)
    # plt.plot(velocity[:,2])
    # plt.title('Front Omega')
    # plt.subplot(233)
    # plt.plot(forces[:,0])
    # plt.title('Front Fx')
    # plt.subplot(234)
    # plt.plot(forces[:,1])
    # plt.title('Front Fy')
    # plt.subplot(235)
    # plt.plot(slips[:,0])
    # plt.title('Front kappa')
    # plt.subplot(236)
    # plt.plot(slips[:,1])
    # plt.title('Front Alpha')
    # plt.figure()
    # plt.plot(ttc)
    # plt.show()
