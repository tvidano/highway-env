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
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from sb3_contrib import TQC
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
    #env = highway_env.envs.collision_env.CollisionEnv()
    env = gym.make('collision-v0')
    
    # Recommended Environment Hypertuning Parameters:
    # env.configure({
    #     "duration": 15,  # [s]
    #     "road_friction": 1.0,
    #     "stopping_vehicles_count": 5,
    #     "time_to_intervene": 6, # [s]
    #     "time_after_collision": 0, # [s]
    #     "vehicles_density": 2,
    #     "vehicles_count": 25,
    #     "control_time_after_avoid": 4, #[s]
    #     "imminent_collision_distance": 7,  # [m]
    #     "reward_type": "penalty_dense"
    # })

    # Uncomment to check environment with OpenAi Gym:
    # check_env(env)


    # Batch simulation parameters
    totalruns = 1000  # number of runs, obviously
    render_env = True  # whether to render the car
    report_every = 1  # how often to report running progress Ex. every 5th run
    do_training = False # whether to train a new model or use a saved one
    model_name = 'PPO' # choose from:  'baseline' = deterministic hard braking, no steering always
                                        #   'PPO' = implements trained PPO if available, otherwise trains a PPO
                                        #   'A2C' = implements trained A2C if available, otherwise trains an A2C
    from_zoo = True # modifier becomes version number

    debug = False # runs only 1 episode and plots outputs on baseline policy

    model_path = model_name.lower() + "_collision"
    modifier = '109'
    model_path += modifier
    if from_zoo:
        model_path = r"../../rl-baselines3-zoo/logs/" + model_name.lower() + "/collision-v0_" + modifier + "/collision-v0.zip"

    reward_stats = []
    num_mitigated = 0
    num_crashed = 0
    num_no_interaction = 0
    num_offroad = 0

    if debug:
        #model_name = 'baseline'
        totalruns = 1
        do_training = False

    if model_name == 'baseline':
        model = None

    elif model_name == 'PPO':
        model = PPO("MlpPolicy", env, learning_rate=0.003, n_steps=2048, batch_size=64, n_epochs=20, verbose=1, device='cuda')
        if do_training:
            print("Training " + model_path)
            start = timeit.default_timer()
            model.learn(total_timesteps=100000, )
            model.save(model_path)
            stop = timeit.default_timer()
            print("Training took", stop-start, "seconds.")
        model.load(model_path)
        print(f'Loaded {model_name} from {os.path.join(os.getcwd(), model_path +".zip")}\n')

    elif model_name == 'A2C':
        model = A2C("MlpPolicy", env, learning_rate=0.003, n_steps=2048,verbose=1)
        if do_training:
            start = timeit.default_timer()
            model.learn(total_timesteps=50000, )
            model.save(model_path)
            stop = timeit.default_timer()
            print("Training took", stop - start, "seconds.")
        model.load(model_path)
        print(f'Loaded {model_name} from {os.path.join(os.getcwd(), model_path +".zip")}\n')

    elif model_name == 'DDPG':
        model = DDPG("MlpPolicy", env, learning_rate=0.003, batch_size=100, verbose=1)
        if do_training:
            start = timeit.default_timer()
            model.learn(total_timesteps=50000, )
            model.save(model_path)
            stop = timeit.default_timer()
            print("Training took", stop - start, "seconds.")
        model.load(model_path)
        print(f'Loaded {model_name} from {os.path.join(os.getcwd(), model_path + ".zip")}\n')

    elif model_name == 'SAC':
        model = SAC("MlpPolicy", env, learning_rate=0.003, batch_size=256, verbose=1)
        if do_training:
            start = timeit.default_timer()
            model.learn(total_timesteps=500000, )
            model.save(model_path)
            stop = timeit.default_timer()
            print("Training took", stop - start, "seconds.")
        model.load(model_path)
        print(f'Loaded {model_name} from {os.path.join(os.getcwd(), model_path + ".zip")}\n')

    elif model_name == 'TD3':
        model = TD3("MlpPolicy", env, learning_rate=0.003, batch_size=100, verbose=1)
        if do_training:
            start = timeit.default_timer()
            model.learn(total_timesteps=50000, )
            model.save(model_path)
            stop = timeit.default_timer()
            print("Training took", stop - start, "seconds.")
        model.load(model_path)
        print(f'Loaded {model_name} from {os.path.join(os.getcwd(), model_path + ".zip")}\n')

    elif model_name == 'TQC':
        model = TQC("MlpPolicy", env, learning_rate=0.003, batch_size=256, verbose=1)
        if do_training:
            start = timeit.default_timer()
            model.learn(total_timesteps=50000, )
            model.save(model_path)
            stop = timeit.default_timer()
            print("Training took", stop - start, "seconds.")
        model.load(model_path)
        print(f'Loaded {model_name} from {os.path.join(os.getcwd(), model_path + ".zip")}\n')

    print("Model", model_name, "trained/loaded")

    previous_run = timeit.default_timer()
    for i in range(totalruns):

        if report_every and i%report_every==0:
            print("Run number ", i)

        obs = env.reset()
        if render_env:
            env.render()
        if debug:
            times = []
            actions = []
            actuators = []
            model_params = []
            velocity = []
            forces = []
            slips = []
            ttc = []
        rewards = []
        info = None
        end_state = 'none'
        done = False
        while not done:
          
            if model:
                action, _states = model.predict(obs)
            else:
                # Use just hard braking (will probably lock the wheel):

                action = np.array([-1, 0.0])

            obs, rew, done, info = env.step(action)
            rewards.append(rew)
            if debug: 
                times.append(info["time"])
                actions.append(info["action"])
                actuators.append(info["actuators"])
                velocity.append(info["speed"])
                forces.append(info["tire_forces"])
                slips.append(info["slip_values"])
                ttc.append(info["ttc"])

            # check if crash has occured or collision has been detected
            if info["imminent"]:
                end_state = "mitigated"
            if info["crashed"]:
                end_state = "crashed"
            if not info['onroad'] and done:
                end_state = 'offroad'

        env.close()

        reward_stats.append(sum(rewards))
        num_no_interaction += end_state == 'none'
        num_crashed += end_state == 'crashed'
        num_mitigated += end_state == 'mitigated'
        num_offroad += end_state == 'offroad'

        if report_every and i%report_every == 0:
            print(sum(rewards))
            print(end_state)
            this_run = timeit.default_timer()
            print('Average time per 1 simulation: ', (this_run - previous_run)/report_every, "seconds.")
            previous_run = this_run


    print(f"Using:\t\t\t{model_name}")
    print(f"Episodes:\t\t{totalruns}")
    print(f"Average reward:\t\t{sum(reward_stats)/len(reward_stats)}")
    print(f"Dummy Episodes:\t\t{num_no_interaction}") # Episodes where no engagements occur
    print(f"Crashes:\t\t{num_crashed}")
    print(f"Offroad:\t\t{num_offroad}")
    print(f"Collisions avoided:\t{num_mitigated}")
    print(f"Avoidance rate:\t\t{num_mitigated/(num_crashed+num_mitigated+num_offroad)*100}%")

    if debug: 
        velocity = np.vstack(velocity)
        forces = np.vstack(forces)
        slips = np.vstack(slips)
        actions = np.vstack(actions)
        actuators = np.vstack(actuators)
        plt.figure()
        plt.subplot(231)
        plt.plot(times, velocity[:,0],'-o',markersize=4)
        plt.title('Long. Vel.')
        plt.subplot(232)
        plt.plot(times, velocity[:,2],'-o',markersize=4)
        plt.title('Front Omega')
        plt.subplot(233)
        plt.plot(times, forces[:,0],'-o',markersize=4)
        plt.title('Front Fx')
        plt.subplot(234)
        plt.plot(times, forces[:,1],'-o',markersize=4)
        plt.title('Front Fy')
        plt.subplot(235)
        plt.plot(times, slips[:,0],'-o',markersize=4)
        plt.title('Front kappa')
        plt.subplot(236)
        plt.plot(times, slips[:,1],'-o',markersize=4)
        plt.title('Front Alpha')
        plt.figure()
        plt.subplot(231)
        plt.plot(times, ttc,'-o',markersize=4)
        plt.title('Time to Collision')
        plt.subplot(232)
        plt.plot(times, actions[:,0],'-o',markersize=4)
        plt.title('Throttle Action')
        plt.subplot(233)
        plt.plot(times, actions[:,1],'-o',markersize=4)
        plt.title('Steering Action')
        plt.subplot(234)
        plt.plot(times, actuators[:, 0],'-o',markersize=4)
        plt.title('Throttle Actuation')
        plt.subplot(235)
        plt.plot(times, actuators[:,1]*180/np.pi,'-o',markersize=4)
        plt.title('Steering Actuation')
        plt.subplot(236)
        plt.plot(times, rewards,'-o',markersize=4)
        plt.title('Rewards')
        plt.show()

