"""
Simulation script for an experiment of adaptive sensing 
applied to autonomous vehicles.
"""

# Environment
import gym # 0.21.0
import sys
import os
# use local version of highway_env, before simulation you should be told this
# is a local version.
local_highway_env = os.path.join(os.getcwd(),"..",)
sys.path.insert(1, local_highway_env)
import highway_env
from gym.wrappers import RecordVideo, RecordEpisodeStatistics

# Agent
from rl_agents.agents.common.factory import agent_factory

# Visualisation
import sys
scripts_path = os.path.join(os.getcwd(),"highway-env","scripts",)
sys.path.insert(0, scripts_path)

# Debugging Tools (uncomment to not show debugging logs)
# import logging
# logging.basicConfig(level=logging.warning)

# Make environment (experimenting right now)
env = gym.make("highway-fast-v0") # we will likely be changing the config of this env.
f_policy = .5
f_sim = 10 * f_policy
t_duration = 30 * f_policy
env.configure({
    "observation": {
            "type": "LidarObservation"
            },
    "simulation_frequency": f_sim, # [Hz]
    "policy_frequency": f_policy,  # [Hz]
    "duration": t_duration,  # [steps per episode]
    "vehicles_density": 1.5,
    "show_trajectories": False,            
    "right_lane_reward": 0.0,  # The reward received when driving on the right-most lanes, linearly mapped to
                               # zero for other lanes.            
    "high_speed_reward": 0.0,  # The reward received when driving at full speed, linearly mapped to zero for
                               # lower speeds according to config["reward_speed_range"].
    "collision_reward": 0.0,    # The reward received when colliding with a vehicle.
    })
#env = RecordVideo(env, "videos")
obs, done = env.reset(), False

# We will likely have to change the reward function. The reward function 
# currently in place encourages maintaining a high speeds, staying in the 
# right lane, and not colliding with other vehicles. We might need to make the
# collision reward more based on distance between other vehicles to encourage
# obstacle avoidance more.

# Make agent
agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "display_tree": True,
    "budget": 20,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

# We currently have a problem with using this planner. It appears that it does
# not rely on observations. Instead it uses the environment to predict 
# what will happen in the future by expanding each node with all possible
# actions. The value appears to be independent of the observations, and so the
# choice of actions is also independent of observations. What we will have to
# do is change the reward function so that it is dependent on observations.
# One way to do this is change the reward function so that it is a function
# of the euclidean distance between the ego vehicle and the surrounding 
# vehicles. Since the surrounding vehicles will be detected by lidar points
# we will actually find the closest lidar point within the ego-vehicle's lane
# and then use that as a cost function. We can also include a fast-speed
# reward to encourage the vehicle not to just come to a stop, but to navigate
# traffic efficiently.

##############################################################################
# Start experiment script here
##############################################################################
# (1) Create an experiment that runs X episodes on a control group: where sensor
#   frequency is unmodified, and the test group: where the sensor frequency is
#   modified according to the discrete reactive scheme discussed in the 
#   proposal.
# (2) Collect data on those 2*X episodes and perform a comparitive analysis to
#   determine if the test group improves in (a) avoiding collisions, (b) 
#   maximizing the reward function, and (c) reducing latency.

# One major issue is recording latency. Unfortunately, this simulator is not
# accurate enough to be a predictor of latency. We can compare how fast the
# simulations run, but this may not correspond to the latency if the system
# were physically implemented. This is because the processing of lidar data
# is not modeled. In this simulation we assume lidar is perfect and tells us
# where other vehicles on the road are. What we can do is record the number of
# lidar samples taken. This way we can compare the number of lidar samples
# taken and the resulting performance.

# Run a single episode
#env.start_video_recorder()
while not done:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
#env.close_video_recorder()

# Begin control group simulation here:
# env = gym.make("highway-lidar-v0")
# env.configure({
#             "adaptive_observations": False,
#             })
# num_episodes = 100
# for i in num_episodes:
#     # Add data collection here
#     done = False
#     while not done:
#         action = agent.act(obs)
#         obs, reward, done, info = env.step(action)
#     env.close()
    
# Begin test group simulation here:
# Need to figure out how to implement the discrete reactive sensor sampling
# scheme with the existing planner.
# env = gym.make("highway-lidar-v0")
# for i in num_episodes:
#     # Add data collection here
#     done = False
#     while not done:
#         action = agent.act(obs)
#         obs, reward, done, info = env.step(action)
#     env.close()

# Begin data analysis here:
# Maybe a bar chart comparing the three metrics we are looking?