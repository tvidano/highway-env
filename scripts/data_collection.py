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
import multiprocessing
from tqdm import tqdm
# use local version of highway_env, before simulation you should be told this
# is a local version.
local_highway_env = op.join(op.dirname(op.realpath(__file__)), "..",)
sys.path.insert(1, local_highway_env)
import highway_env  # noqa
from highway_env.data.markov_chain import discrete_markov_chain  # noqa


logger.setLevel(logger.INFO)


def convert_array_to_int(array):
    output = 0
    for i, binary_digit in enumerate(array):
        output += binary_digit * 2**i
    return int(output)


def convert_int_to_array(int):
    return f'{int:016b}'


def experiment(start_seed, end_seed, shared_dict):
    # Create and configure gym environment.
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

    # start_seed = 100_000
    # end_seed = 100_010
    exp_record = []
    for seed in tqdm(range(start_seed, end_seed + 1)):
        print(f"seed:{seed}")
        terminated, truncated = False, False
        obs = env.reset(seed=seed)
        state_record = []
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
            state_record.append(obs_state)
            # env.render()
        shared_dict[seed] = state_record
        # exp_record.append(state_record)
    # print(f"unique states:\t{np.unique(exp_record)}")
    # mc = discrete_markov_chain(
    #     transition_data=exp_record, num_states=num_states)
    # print(f"irreducible? {mc.is_irreducible()}")
    # mc.save_object(op.join(".", f"{start_seed}_{end_seed}"))
    env.close()


if __name__ == "__main__":
    with multiprocessing.Manager() as manager:
        # Define data collection configuration.
        start_seed = 2_000
        end_seed = 3_000
        num_processes = 5
        step = int((end_seed - start_seed) / num_processes)

        # Get ranges for each process.
        start_seeds = np.arange(start_seed, end_seed, step)
        end_seeds = np.arange(start_seed - 1 + step, end_seed, step)
        assert len(start_seeds) == len(end_seeds) == num_processes
        # Create shared dictionary.
        experiment_record = manager.dict()
        # Create processes.
        processes = [
            multiprocessing.Process(
                target=experiment,
                args=(start, end, experiment_record))
            for start, end in zip(start_seeds, end_seeds)]
        # Start all processes.
        for process in processes:
            process.start()
        # Wait for all processes to complete.
        for process in processes:
            process.join()
        experiment_data = list(experiment_record.values())
        mc = discrete_markov_chain(
            transition_data=experiment_data, num_states=2**16)
        mc.save_object(
            f"2_lane_low_density_low_cars_{min(start_seeds)}_{max(end_seeds)}")
        print(experiment_record)
