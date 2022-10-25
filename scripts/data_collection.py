"""
Simulation script for an experiment of adaptive sensing 
applied to autonomous vehicles.
"""

# Environment
from rl_agents.agents.common.factory import agent_factory
from rl_agents.trainer.evaluation import Evaluation

import gym  # 0.26.2
from gym import logger
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

# Run a single episode:
logger.setLevel(logger.INFO)
env = gym.make("highway-lidar-v0")
# env = RecordVideo(env, "videos")
env.configure({
    "adaptive_observations": False,
    "constant_base_lidar": True,
    "base_lidar_frequency": 1.0,
})

# Make agent
agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method": "simplify"}],
    "display_tree": False,
    "budget": 50,
    "gamma": 0.9,
}
agent = agent_factory(env, agent_config)

# env.start_video_recorder()
action_dict = env.action_type.ACTIONS_ALL
terminated, truncated = False, False
obs = env.reset(seed=111_111)
lidar_count = 0
while not truncated and not terminated:
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    #print(f"last Observation: {obs}")
    closest_car = env.find_closest_obstacle()
    print(f"{env.lidar_count - lidar_count} new lidar points")
    lidar_count = env.lidar_count
    print(f"Closest: \t {closest_car:.2f} m away.")
    print(
        f"action: \t {action_dict[action]} \t vel: {env.vehicle.speed:.1f} tar.vel.: {env.vehicle.target_speed:.1f}")
    print(f"reward: \t {reward:.2f}")
    print(f'\t \t {env.get_reward_breakdown(action)}')
    print(f"crashed: \t{env.vehicle.crashed}")
    # Index is clockwise.
    print(f"updated indexes: {env.indexes_to_update}")
    print(f"lidar: \t {env.lidar_buffer[:,0]}")
    print(f"time: \t{env.time}")
    env.render()
env.close()
