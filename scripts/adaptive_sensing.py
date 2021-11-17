"""
Simulation script for an experiment of adaptive sensing 
applied to autonomous vehicles.
"""

# Environment
import gym # 0.21.0
import sys
import os.path as op
# use local version of highway_env, before simulation you should be told this
# is a local version.
local_highway_env = op.join(op.dirname(op.realpath(__file__)),"..",)
sys.path.insert(1, local_highway_env)
import highway_env
from gym.wrappers import RecordVideo, RecordEpisodeStatistics
import time

# Agent
from rl_agents.agents.common.factory import agent_factory

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
# f_policy = .5 
# f_sim = 10 * f_policy
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
#     "right_lane_reward": 0.0,  # The reward received when driving on the right-
#                                # most lanes, linearly mapped to zero for other
#                                # lanes.            
#     "high_speed_reward": 0.4,  # The reward received when driving at full 
#                                # speed, linearly mapped to zero for lower
#                                # speeds according to
#                                # config["reward_speed_range"].
#     "collision_reward": -0.5,   # The reward received when colliding with a 
#                                # vehicle.
#     "lane_change_reward": 0.4,
#     })
# #env = RecordVideo(env, "videos") 
# obs, done = env.reset(), False

# # Make agent
# agent_config = {
#     "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
#     "env_preprocessors": [{"method":"simplify"}],
#     "display_tree": True,
#     "budget": 20,
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
# # Run a single episode:
# env = gym.make("highway-lidar-v0")
# # env = RecordVideo(env, "videos") 
# env.configure({
#             "adaptive_observations": False,
#             })

# # Make agent
# agent_config = {
#     "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
#     "env_preprocessors": [{"method":"simplify"}],
#     "display_tree": True,
#     "budget": 50,
#     "gamma": 0.9,
# }
# agent = agent_factory(env, agent_config)

# # env.start_video_recorder()
# action_dict = env.action_type.ACTIONS_ALL
# done = False
# obs = env.reset()
# while not done:
#     action = agent.act(obs)
#     obs, reward, done, info = env.step(action)
#     #print(f"last Observation: {obs}")
#     closest_car = env.find_closest_obstacle()
#     print(f"Closest {closest_car:.2f} m away.")
#     print(f"action {action_dict[action]}")
#     print(f"reward {reward:.2f}")
#     env.render()
# env.close()
# # env.close_video_recorder()

###############################################################################
# Begin control group simulation here:
env = gym.make("highway-lidar-v0")
env.configure({
            "adaptive_observations": False,
            })

# Make agent for control group
agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "display_tree": True,
    "budget": 50,
    "gamma": 0.9,
}
agent = agent_factory(env, agent_config)

# Create statistics variables
control_collisions = 0
control_rewards = []
control_lidar_samples = []

num_episodes = 100
control_start_time = time.time()
for i in range(num_episodes):
    # Add data collection here
    done = False
    episode_reward = 0
    obs = env.reset()
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    control_collisions += float(env.vehicle.crashed)
    control_rewards.append(episode_reward)
    control_lidar_samples.append(env.lidar_count)
    env.close()
    print(f"Completed control episode {i}")
control_end_time = time.time()

###############################################################################
# Begin test group simulation here:
env = gym.make("highway-lidar-v0")
env.configure({
            "adaptive_observations": True,
            })

# Make agent for control group
agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "display_tree": True,
    "budget": 50,
    "gamma": 0.9,
}
agent = agent_factory(env, agent_config)

# Create statistics variables
test_collisions = 0
test_rewards = []
test_lidar_samples = []

test_start_time = time.time()
for i in range(num_episodes):
    # Add data collection here
    done = False
    episode_reward = 0
    obs = env.reset()
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    test_collisions += float(env.vehicle.crashed)
    test_rewards.append(episode_reward)
    test_lidar_samples.append(env.lidar_count)
    env.close()
    print(f"Completed test episode {i}")
test_end_time = time.time()

###############################################################################
# Begin data analysis here:
# Maybe a bar chart comparing the three metrics we are looking?
print(f"control collision: {control_collisions}")
print(f"control rewards: {control_rewards}")
print(f"control lidar samples: {control_lidar_samples}")
print(f"test collision: {test_collisions}")
print(f"test rewards: {test_rewards}")
print(f"test lidar samples: {test_lidar_samples}")

print(f"ave. control rew: {sum(control_rewards)/len(control_rewards)}")
print(f"ave. test rew: {sum(test_rewards)/len(test_rewards)}")
print(f"ave. control lidar samples: {sum(control_lidar_samples)/len(control_lidar_samples)}")
print(f"ave. test lidar samples: {sum(test_lidar_samples)/len(test_lidar_samples)}")
print(f"control runtime: {control_end_time - control_start_time}")
print(f"test runtime: {test_end_time - test_start_time}")