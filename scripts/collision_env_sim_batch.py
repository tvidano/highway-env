# Environment
import gym
import sys
import timeit

# Models and computation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import A2C
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
import os
import time

if 'highway_env' not in sys.modules:
    try:
        import highway_env
    except ImportError:
        sys.path.append(os.getcwd())
        import highway_env

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    env = highway_env.envs.collision_env.CollisionEnv()

    # Recommended Environment Hypertuning Parameters:
    # env.configure({
    #     "duration": 10,  # [s]
    #     "road_friction": 1.0,
    #     "stopping_vehicles_count": 5,
    #     "time_to_intervene": 2, # [s]
    #     "time_after_collision": 0, # [s]
    #     "vehicles_density": 2,
    #     "vehicles_count": 20,
    # })

    # Uncomment to check environment with OpenAi Gym:
    # check_env(env)

    # Batch simulation parameters
    totalruns = 100  # number of runs, obviously
    render_env = True  # whether to render the car
    report_every = 10  # how often to report running progress Ex. every 5th run
    model_name = 'ppo' # choose from:  'baseline' = deterministic hard braking, no steering always
                                        #   'ppo' = implements trained PPO if available, otherwise trains a PPO
                                        #   'a2c' = implements trained A2C if available, otherwise trains an A2C
    model_path = model_name.lower() + "_collision"

    # Statistics portion
    reward_stats = []
    num_mitigated = 0
    num_crashed = 0
    num_no_interaction = 0

    if model_name == 'baseline':
        model = None
    else:
        if model_name == 'ppo':
            model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=2048,
                        batch_size=64, n_epochs=10,verbose=1)
        elif model_name == 'a2c':
            model = A2C("MlpPolicy", env, learning_rate=0.0003, n_steps=2048,verbose=1)

        try:
            model.load(model_path)
            print(f'Loaded {model_name} from {os.path.join(os.getcwd(), model_path +".zip")}\n')
        except FileNotFoundError:
            print(f'Unable to find {os.path.join(os.getcwd(),model_path)} training new model...\n')
            model.learn(total_timesteps=10000, )   
            model.save(model_name.lower() + "_collision")

    previous_run = timeit.default_timer()
    for i in range(totalruns):

        if report_every and i%report_every==0:
            print("Run number ", i)

        obs = env.reset()
        if render_env:
            env.render()
        times = []
        model_params = []
        rewards = []
        velocity = []
        forces = []
        slips = []
        ttc = []
        end_state = 'none'
        done = False
        while not done:
            if model:
                action, _states = model.predict(obs)
            else:
                # Use just hard braking (will probably lock the wheel):
                action = np.array([-1, 0.0])

            obs, rew, done, info = env.step(action)
            times.append(info["time"])
            velocity.append(info["speed"])
            forces.append(info["tire_forces"])
            slips.append(info["slip_values"])
            ttc.append(info["ttc"])
            rewards.append(rew)

            # check if crash has occured or collision has been detected
            if info["imminent"]:
                end_state = "mitigated"
            if info["crashed"]:
                end_state = "crashed"

        env.close()

        reward_stats.append(rewards[-1])
        num_no_interaction += end_state == 'none'
        num_crashed += end_state == 'crashed'
        num_mitigated += end_state == 'mitigated'

        if report_every and i%report_every == 0:
            print(end_state)
            this_run = timeit.default_timer()
            print('Average time per 1 simulation: ', (this_run - previous_run)/report_every)
            previous_run = this_run


    print("Using: ", model_name)
    print("Total runs: ", totalruns)
    print("Average reward: ", sum(reward_stats)/len(reward_stats))
    print("Number of runs without intervention needed (dummy runs): ", num_no_interaction)
    print("Number of runs with crashes: ", num_crashed)
    print("Number of runs with collisions avoided: ", num_mitigated)
    print("Success rate at avoiding collisions: ", num_mitigated/(num_crashed+num_mitigated), '%')

    # Uncomment to see plots of velocities + forces + slippage
    # velocity = np.vstack(velocity)
    # forces = np.vstack(forces)
    # slips = np.vstack(slips)
    # print(rewards)
    # plt.figure()
    # plt.subplot(231)
    # plt.plot(times, velocity[:,0])
    # plt.title('Long. Vel.')
    # plt.subplot(232)
    # plt.plot(times, velocity[:,2])
    # plt.title('Front Omega')
    # plt.subplot(233)
    # plt.plot(times, forces[:,0])
    # plt.title('Front Fx')
    # plt.subplot(234)
    # plt.plot(times, forces[:,1])
    # plt.title('Front Fy')
    # plt.subplot(235)
    # plt.plot(times, slips[:,0])
    # plt.title('Front kappa')
    # plt.subplot(236)
    # plt.plot(times, slips[:,1])
    # plt.title('Front Alpha')
    # plt.figure()
    # plt.plot(times, ttc)
    # plt.show()

