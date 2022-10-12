"""
Simulation script for an experiment of adaptive sensing 
applied to autonomous vehicles.
"""

# Environment
from rl_agents.agents.common.factory import agent_factory
from logging import log
import gym  # 0.21.0
from gym.wrappers import RecordVideo, RecordEpisodeStatistics
import os
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
# use local version of highway_env, before simulation you should be told this
# is a local version.
local_highway_env = op.join(op.dirname(op.realpath(__file__)), "..",)
sys.path.insert(1, local_highway_env)
import highway_env  # noqa

# Agent

# Debugging Tools (uncomment to not show debugging logs)
# import logging
# logging.basicConfig(level=logging.DEBUG)

###############################################################################
# To get started, comment out everything below this section and play around
# with this code. You should dive deeper into the code to understand how the
# original reward function affects the ego-vehicle's behavior.
# Once you're done playing around and understand somewhat of how the simulation
# is structured, comment out this part so that you can run the simulations.
###############################################################################
# # Make environment (experimenting right now)
# env = gym.make("highway-fast-v0")
# f_policy = 1
# f_sim = 5 * f_policy
# t_duration = 30 * f_policy
# env.configure(
#     {"observation":
#         {"type":
#             "LidarObservation"
#         },
#     "simulation_frequency": f_sim, # [Hz]
#     "policy_frequency": f_policy,  # [Hz]
#     "duration": t_duration,  # [steps per episode]
#     "vehicles_density": 1.5,
#     "show_trajectories": False,
#     # "right_lane_reward": 0.0,  # The reward received when driving on the right-
#     #                            # most lanes, linearly mapped to zero for other
#     #                            # lanes.
#     # "high_speed_reward": 0.4,  # The reward received when driving at full
#     #                            # speed, linearly mapped to zero for lower
#     #                            # speeds according to
#     #                            # config["reward_speed_range"].
#     # "collision_reward": -0.5,   # The reward received when colliding with a
#     #                            # vehicle.
#     # "lane_change_reward": 0.4,
#     })
# #env = RecordVideo(env, "videos")
# obs, done = env.reset(), False

# # Make agent
# agent_config = {
#     "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
#     "env_preprocessors": [{"method":"simplify"}],
#     "display_tree": True,
#     "budget": 50,
#     "gamma": 0.7,
# }
# agent = agent_factory(env, agent_config)

# while not done:
#     action = agent.act(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()
# env.close()

###############################################################################
# Proposed simulation design
###############################################################################
# This planner does not rely on observations. Instead, it uses copies of the
# environment to predict what will happen in the future by stepping forward in
# time with all possible actions. The value of each state is independent of the
# observations, because the reward function does not depend on the
# observations. What we will have to do is change the reward function so that
# it is dependent on observations.
# I think the easiest way to do this is to punish the closest distance between
# the ego-vehicle and a buffered lidar point. The reward should also only care
# about the lidar points within its own lane. If we consider lidar points in
# other lanes then the ego-vehicle will steer away from vehicles in the
# neighboring lane. We should also reward maintaining a fast-speed to encourage
# the vehicle not to just come to a stop, but to navigate traffic efficiently.
# The key aspect of this is that the reward is based on 'buffered' lidar
# points. This means that the lidar points need to be stored so that they can
# be observered asynchronously (at different sampling rates). This will allow
# us to create the adaptive sampling controller discussed in the paper.

###############################################################################
# Start experiment script here
###############################################################################
# High-level project tasks.
# (1) Create an experiment that runs X episodes on a control group: where
#   sensor frequency is unmodified, and the test group: where the sensor
#   frequency is modified according to the discrete reactive scheme discussed
#   in the proposal.
# (2) Collect data on those 2*X episodes and perform a
#   comparitive analysis to determine if the test group improves in (a)
#   avoiding collisions, (b) maximizing the reward function, and (c) reducing
#   latency (as measured by lidar).

# One major issue is recording latency. Unfortunately, this simulator is not
# accurate enough to be a predictor of latency. We can compare how fast the
# simulations run, but this may not correspond to the latency if the system
# were physically implemented. This is because the processing of lidar data
# is not modeled. In this simulation we assume lidar is perfect and tells us
# where other vehicles on the road are. What we can do is record the number of
# lidar samples taken. This way we can compare the number of lidar samples
# taken and the resulting performance.

###############################################################################
# Run a single episode:
env = gym.make("highway-lidar-v0")
# env = RecordVideo(env, "videos")
env.configure({
    "adaptive_observations": False,
    "constant_base_lidar": True,
})

# Make agent
agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method": "simplify"}],
    "display_tree": True,
    "budget": 100,
    "gamma": 0.9,
}
agent = agent_factory(env, agent_config)

# env.start_video_recorder()
action_dict = env.action_type.ACTIONS_ALL
done = False
obs = env.reset()
lidar_count = 0
while not done:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    #print(f"last Observation: {obs}")
    closest_car = env.find_closest_obstacle()
    print(f"{env.lidar_count - lidar_count} new lidar points")
    lidar_count = env.lidar_count
    print(f"Closest: \t {closest_car:.2f} m away.")
    print(f"action: \t {action_dict[action]}")
    print(f"reward: \t {reward:.2f}")
    print(f"crashed: \t{env.vehicle.crashed}")
    print(f"time: \t{env.current_time}")
    env.render()
env.close()
# env.close_video_recorder()

# ###############################################################################
# # Create experiment log file:
# log_dir = op.join(local_highway_env, "scripts","experiment_4")
# os.makedirs(log_dir, exist_ok=True)
# log_file = op.join(log_dir, "experiment.log")
# raw_control_data = op.join(log_dir, "control_raw.log")
# raw_test_data = op.join(log_dir, "test_raw.log")
# agent_budget = 50
# agent_gamma = 0.9
# f_sim = 5
# f_policy = 1
# f_lidar = 0.25
# num_episodes = 500

# # Begin control group 1 simulation here:
# env = gym.make("highway-lidar-v0")
# env.configure({
#             "adaptive_observations": False,
#             "simulation_frequency": f_sim,
#             "policy_frequency": f_policy,
#             "base_lidar_frequency": f_lidar,  # <= policy_frequency
#             "duration": 30 * f_policy,
#             "constant_base_lidar": False,
#             })

# # Make agent for control groups
# agent_config = {
#     "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
#     "env_preprocessors": [{"method":"simplify"}],
#     "display_tree": False,
#     "budget": agent_budget,
#     "gamma": agent_gamma,
# }
# agent = agent_factory(env, agent_config)

# with open(log_file, 'w') as f:
#     print("*" * 79, file=f)
#     print("Experimental Setup", file=f)
#     print(f"Simulation Frequency: {f_sim}",file=f)
#     print(f"Policy Frequency: {f_policy}", file=f)
#     print(f"Lidar Frequency: {f_lidar}", file=f)
#     print(f"Number of Episodes: {num_episodes}", file=f)
#     print("*" * 79, file=f)
#     print("*" * 79, file=f)
#     print("Control Group 1", file=f)
#     print(f"Agent Budget: {agent_budget}", file=f)
#     print(f"Agent Gamma: {agent_gamma}", file=f)
#     print(f"Sensing Policy: Constant at {f_policy} Hz", file=f)
#     print("*" * 79, file=f)

# # Create statistics variables
# control_collisions = []
# control_rewards = []
# control_lidar_samples = []
# control_times = []

# control_start_time = time.time()
# with open(log_file, 'a') as f:
#     print("Episode # | Total Rew. | If Crashed | Lidar Samples", file=f)
# for i in range(num_episodes):
#     # Add data collection here
#     done = False
#     episode_reward = 0
#     obs = env.reset()
#     while not done:
#         action = agent.act(obs)
#         obs, reward, done, info = env.step(action)
#         episode_reward += reward
#     control_collisions.append(float(env.vehicle.crashed))
#     control_rewards.append(episode_reward)
#     control_lidar_samples.append(env.lidar_count)
#     control_times.append(env.steps / f_policy)
#     env.close()
#     with open(log_file, 'a') as f:
#         print(f"{i},{episode_reward:.2f},{env.vehicle.crashed},{env.lidar_count}", file=f)
#     print(f"Completed control episode {i}")
# control_end_time = time.time()

# # Begin control group 2 simulation here:
# env = gym.make("highway-lidar-v0")
# env.configure({
#             "adaptive_observations": False,
#             "simulation_frequency": f_sim,
#             "policy_frequency": f_policy,
#             "base_lidar_frequency": f_lidar,  # <= policy_frequency
#             "duration": 30 * f_policy,
#             "constant_base_lidar": True,
#             })

# with open(log_file, 'a') as f:
#     print("*" * 79, file=f)
#     print("Control Group 2", file=f)
#     print(f"Agent Budget: {agent_budget}", file=f)
#     print(f"Agent Gamma: {agent_gamma}", file=f)
#     print(f"Sensing Policy: Constant at {f_lidar} Hz", file=f)
#     print("*" * 79, file=f)

# # Make agent for control groups
# agent_config = {
#     "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
#     "env_preprocessors": [{"method":"simplify"}],
#     "display_tree": False,
#     "budget": agent_budget,
#     "gamma": agent_gamma,
# }
# agent = agent_factory(env, agent_config)

# # Create statistics variables
# control2_collisions = []
# control2_rewards = []
# control2_lidar_samples = []
# control2_times = []

# control2_start_time = time.time()
# with open(log_file, 'a') as f:
#     print("Episode # | Total Rew. | If Crashed | Lidar Samples", file=f)
# for i in range(num_episodes):
#     # Add data collection here
#     done = False
#     episode_reward = 0
#     obs = env.reset()
#     while not done:
#         action = agent.act(obs)
#         obs, reward, done, info = env.step(action)
#         episode_reward += reward
#     control2_collisions.append(float(env.vehicle.crashed))
#     control2_rewards.append(episode_reward)
#     control2_lidar_samples.append(env.lidar_count)
#     control2_times.append(env.steps / f_policy)
#     env.close()
#     with open(log_file, 'a') as f:
#         print(f"{i},{episode_reward:.2f},{env.vehicle.crashed},{env.lidar_count}", file=f)
#     print(f"Completed control 2 episode {i}")
# control2_end_time = time.time()

# ###############################################################################
# # Begin test group simulation here:
# env = gym.make("highway-lidar-v0")
# env.configure({
#             "adaptive_observations": True,
#             "simulation_frequency": f_sim,
#             "policy_frequency": f_policy,
#             "base_lidar_frequency": f_lidar,  # <= policy_frequency
#             })

# # Make agent for test group
# agent_config = {
#     "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
#     "env_preprocessors": [{"method":"simplify"}],
#     "display_tree": True,
#     "budget": agent_budget,
#     "gamma": agent_gamma,
# }
# agent = agent_factory(env, agent_config)

# with open(log_file, 'a') as f:
#     print("*" * 79, file=f)
#     print("Test Group", file=f)
#     print(f"Agent Budget: {agent_budget}", file=f)
#     print(f"Agent Gamma: {agent_gamma}", file=f)
#     print(f"Sensing Policy: Adaptive at {f_lidar} Hz", file=f)
#     print("*" * 79, file=f)

# # Create statistics variables
# test_collisions = []
# test_rewards = []
# test_lidar_samples = []
# test_times = []

# test_start_time = time.time()
# with open(log_file, 'a') as f:
#     print("Episode # | Total Rew. | If Crashed | Lidar Samples", file=f)
# for i in range(num_episodes):
#     # Add data collection here
#     done = False
#     episode_reward = 0
#     obs = env.reset()
#     while not done:
#         action = agent.act(obs)
#         obs, reward, done, info = env.step(action)
#         episode_reward += reward
#     test_collisions.append(float(env.vehicle.crashed))
#     test_rewards.append(episode_reward)
#     test_lidar_samples.append(env.lidar_count)
#     test_times.append(env.steps / f_policy)
#     env.close()
#     with open(log_file, 'a') as f:
#         print(f"{i},{episode_reward:.2f},{env.vehicle.crashed},{env.lidar_count}", file=f)
#     print(f"Completed test episode {i}")
# test_end_time = time.time()

# ###############################################################################
# # Begin data analysis here:
# # Maybe a bar chart comparing the three metrics we are looking?
# labels = ["High Freq. Sensing","Adaptive Sensing","Low Freq. Sensing"]
# x = np.arange(len(labels))
# control_lidar_per_sec = np.array(control_lidar_samples) / np.array(control_times)
# control2_lidar_per_sec = np.array(control2_lidar_samples) / np.array(control2_times)
# test_lidar_per_sec = np.array(test_lidar_samples) / np.array(test_times)
# lidar = [np.mean(control_lidar_per_sec), np.mean(test_lidar_per_sec), \
#          np.mean(control2_lidar_per_sec)]
# lidar_std = [np.std(control_lidar_per_sec), np.std(test_lidar_per_sec), \
#              np.std(control2_lidar_per_sec)]
# crashes = [np.mean(control_collisions), np.mean(test_collisions), \
#            np.mean(control2_collisions)]
# crashes_std = [np.std(control_collisions), np.std(test_collisions), \
#                np.std(control2_collisions)]

# fig1, ax1 = plt.subplots()
# ax1.bar(x, lidar, yerr=lidar_std, alpha=0.5, ecolor='black', capsize=10)
# ax1.set_ylabel('Average Lidar Samples per Second per Episode')
# ax1.set_xticks(x)
# ax1.set_xticklabels(labels)
# ax1.yaxis.grid(True)
# fig1.savefig(op.join(log_dir,"lidar_samples.png"))
# fig1.show()

# fig2, ax2 = plt.subplots()
# ax2.bar(x, crashes, yerr=crashes_std, alpha=0.5, ecolor='black', capsize=10)
# ax2.set_ylabel('Average Collisions per Episode')
# ax2.set_xticks(x)
# ax2.set_xticklabels(labels)
# ax2.yaxis.grid(True)
# plt.ylim([0,1])
# fig2.savefig(op.join(log_dir, "collisions.png"))
# fig2.show()

# with open(log_file, 'a') as f:
#     print("*" * 79, file=f)
#     print("Experiment Summary", file=f)
#     print("*" * 79, file=f)
#     print("*" * 79, file=f)
#     print("Raw Data:", file=f)
#     print(f"control 1 collision: {control_collisions}", file=f)
#     print(f"control 1 rewards: {control_rewards}", file=f)
#     print(f"control 1 lidar samples: {control_lidar_samples}", file=f)
#     print(f"control 2 collision: {control2_collisions}", file=f)
#     print(f"control 2 rewards: {control2_rewards}", file=f)
#     print(f"control 2 lidar samples: {control2_lidar_samples}", file=f)
#     print(f"test collision: {test_collisions}", file=f)
#     print(f"test rewards: {test_rewards}", file=f)
#     print(f"test lidar samples: {test_lidar_samples}", file=f)

#     print("*" * 79, file=f)
#     print("Aggregated Data:", file=f)
#     print(f"total control 1 collisions: {sum(control_collisions)}", file=f)
#     print(f"ave. control 1 rew: {sum(control_rewards)/len(control_rewards)}", file=f)
#     print(f"ave. control 1 lidar samples per sec: {np.mean(control_lidar_per_sec)}", file=f)
#     print(f"std control 1 lidar_samples per sec: {np.std(control_lidar_per_sec)}", file=f)
#     print(f"total control 2 collisions: {sum(control2_collisions)}", file=f)
#     print(f"ave. control 2 rew: {sum(control2_rewards)/len(control2_rewards)}", file=f)
#     print(f"ave. control 2 lidar samples per sec:  {np.mean(control2_lidar_per_sec)}", file=f)
#     print(f"std control 2 lidar samples per sec: {np.std(control2_lidar_per_sec)}", file=f)
#     print(f"total test collisions: {sum(test_collisions)}", file=f)
#     print(f"ave. test rew: {sum(test_rewards)/len(test_rewards)}", file=f)
#     print(f"ave. test lidar samples per sec: {np.mean(test_lidar_per_sec)}", file=f)
#     print(f"std test lidar samples per sec: {np.std(test_lidar_per_sec)}", file=f)

#     print("*" * 79, file=f)
#     print("Runtime Data:", file=f)
#     print(f"control 1 runtime: {control_end_time - control_start_time}", file=f)
#     print(f"control 2 runtime: {control2_end_time - control2_start_time}", file=f)
#     print(f"test runtime: {test_end_time - test_start_time}", file=f)
#     print(f"control 1 simulated time: {sum(control_times)}", file=f)
#     print(f"control 2 simulated time: {sum(control2_times)}", file=f)
#     print(f"test simulated time: {sum(test_times)}", file=f)
