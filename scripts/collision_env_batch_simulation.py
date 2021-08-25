# Environment
import gym
import sys
import timeit

# Models and computation
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from sb3_contrib import TQC

# Visualization
import matplotlib.pyplot as plt

# IO
import os

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

    # Batch simulation parameters
    config = {
        "env_name": "collision-v0",
        "num_runs": 10,
        "render_env": True,         # whether to render the car
        "report_every": 1,        # how often to report running progress Ex. every 5th run
        "do_training": True,       # whether to train a new model or use a saved one
        "model": {
            "name": "PPO",     # choose from:  'baseline' = deterministic hard braking, no steering always
                                    # 'PPO', 'A2C', 'DDPG', 'SAC', 'TD3', 'TQC' to train or load those models
            "identifier": "",    # adds identifier to model name so multiple models can be stored
            "params": {             # parameters to pass to model
                "policy": "MlpPolicy",
                "learning_rate": 0.003,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 20,
                "verbose": 1,
            },
            "n_timesteps": 5000,   # how long to train the model for
            "from_zoo": False,       # identifier becomes rl-baselines-zoo model number and program looks for zoo file in logs folder
            "zoo_path": r"../../rl-baselines3-zoo",  # local path of rl-zoo directory
        },
        "debug": False,             # plots variables & vehicle data after each run
    }



    env = gym.make(config["env_name"])

    model_path = rf"{config['model']['name'].lower()}_collision{config['model']['identifier']}"
    if config["model"]["from_zoo"]:
        model_path = rf"""{config['model']['zoo_path']}/logs/{config['model']['name'].lower()}/{config['env_name']
            }_{config['model']['identifier']}/{config['env_name']}.zip"""

    # Statistics collection variables
    reward_stats = []
    num_mitigated = 0
    num_crashed = 0
    num_no_interaction = 0
    num_offroad = 0

    if config['model']['name'] == 'baseline':
        model = None
    else:
        if config['model']['name'] in ['PPO', 'A2C', 'DDPG', 'SAC', 'TD3', 'TQC'] and globals()[config['model']['name']]:
            model = globals()[config['model']['name']](**config['model']['params'], env=config["env_name"])
        else:
            print("Invalid model name, using baseline")
            model = None

    if model:
        if config["do_training"]:
            print("Training " + model_path)
            start = timeit.default_timer()
            model.learn(total_timesteps=config["model"]["n_timesteps"], )
            model.save(model_path)
            stop = timeit.default_timer()
            print(f"Training took {stop - start} seconds.")
        model.load(model_path)
        print(f"Loaded {config['model']['name']} from {os.path.join(os.getcwd(), model_path + '.zip')}\n")

    print(f"Using {config['model']['name']}, starting run 0...")

    previous_run = timeit.default_timer()
    for i in range(config["num_runs"]):

        if config["report_every"] and (i+1)%config["report_every"]==0:
            print("Run number ", i+1)

        obs = env.reset()
        if config["render_env"]:
            env.render()
        if config["debug"]:
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
                # Use just hard braking (will probably lock the wheel)
                action = np.array([-1, 0.0])

            obs, rew, done, info = env.step(action)

            rewards.append(rew)
            if config["debug"]:
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

        reward_stats.append(sum(rewards))
        num_no_interaction += end_state == 'none'
        num_crashed += end_state == 'crashed'
        num_mitigated += end_state == 'mitigated'
        num_offroad += end_state == 'offroad'

        if config["report_every"] and (i+1)%config["report_every"] == 0:
            print(f"Reward: {sum(rewards)}, End State: {end_state}")
            this_run = timeit.default_timer()
            print(f"Average time per 1 simulation: {(this_run - previous_run)/config['report_every']} seconds.")
            previous_run = this_run


    print(f"Using:\t\t\t{config['model']['name']}")
    print(f"Episodes:\t\t{config['num_runs']}")
    print(f"Average reward:\t\t{sum(reward_stats)/len(reward_stats)}")
    print(f"Dummy Episodes:\t\t{num_no_interaction}")  # Episodes where no engagements occur
    print(f"Crashes:\t\t{num_crashed}")
    print(f"Offroad:\t\t{num_offroad}")
    print(f"Collisions avoided:\t{num_mitigated}")
    print(f"Avoidance rate:\t\t{num_mitigated/(num_crashed+num_mitigated+num_offroad)*100}%")

    if config["debug"]:
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
