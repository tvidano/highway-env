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
import pickle
import sys
import time
from tqdm import tqdm
# use local version of highway_env, before simulation you should be told this
# is a local version.
local_highway_env = op.join(op.dirname(op.realpath(__file__)), "..",)
sys.path.insert(1, local_highway_env)
import highway_env  # noqa
from highway_env.data.markov_chain import markov_chain  # noqa

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


def convert_array_to_int(array):
    output = 0
    for i, binary_digit in enumerate(array):
        output += binary_digit * 2**i
    return int(output)


for seed in tqdm(range(111_110, 111_111)):
    terminated, truncated = False, False
    obs = env.reset(seed=seed)
    num_states = 2**16
    prev_state = 0
    mc = markov_chain(num_states)
    while not truncated and not terminated:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        closest_car = env.find_closest_obstacle()
        # Index is clockwise.
        obs_state = convert_array_to_int(env.lidar_buffer[:, 0] < np.inf)
        # Build markov chain.
        # Assumptions:
        #   * stationarity
        #   * markov property
        #   * time-invariant
        mc.update_transition_matrix(prev_state, obs_state)
        prev_state = obs_state
        # env.render()
    mc.save_object(f"mc_{seed}")
    with open(f"mc_{seed}", "rb") as file:
        mc2 = pickle.load(file)
    mc.get_stationary_matrix()
env.close()
