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

import json
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

# Experiment: 11/15/2022 "4_lane_high_density_high_cars"
# vehicles_density works initially but other vehicles quickly disperse.
# modified IDMVehicle:
#       COMFORT_ACC_MAX = 0.73 (from IDM paper)
#       COMFORT_ACC_MIN = -1.67 (from IDM paper)
#       DISTANCE_WANTED = 2 + ... (guessed from IDM paper)
#       TIME_WANTED = 0.6 (to encourage vehicle density to last longer)
#       LANE_CHANGE_DELAY = 2.0 (collisions were too frequent)
# modified experiment params:
#       duration = 15 (prevent ego_vehicle from leaving all other vehicles)
#       vehicles_density = 3.0 (roughly the max before initializing with too
#                               many collisions)
#       vehicles_count = 30 (prevents ego_vehicle from leaving all other   
#                            vehicles)
# Experiment: 11/17/2022 "2_lane_low_density"
# The frequency of scenes in which the ego vehicle makes it into open road
# is too great.
# modified HighwayEnvLidar to add _cycle_vehicles().
#       this now cycles vehicles whenever a vehicle passes the ego-vehicle
#       or whenever the ego-vehicle passes. 
# experiment params:
#       lanes_count = 2
#       duration = 30
#       vehicles_density = 1
#       vehicles_count = 10
# 
# Experiment: 11/21/2022 "4_lane_low_density"
# experiment params:
#       lanes_count = 4
#       duration = 30
#       vehicles_density = 1
#       vehicles_count = 10
#
def experiment(start_seed, end_seed, config, state_record, 
               ego_collisions_record, vehicles_count_record, 
               environment_config, lane_changes_record, avg_velocity):
    # Create and configure gym environment.
    env = gym.make("highway-lidar-v0")
    # env = RecordVideo(env, "videos")
    env.configure({
        "adaptive_observations": False,
        "constant_base_lidar": True,
        "base_lidar_frequency": 1.0,
        **{k: config["experiment"][k] for k in config["experiment"].keys()},
        # "lanes_count": config["experiment"]["lanes_count"],
        # "vehicles_count": config["experiment"]["vehicles_count"],
        # "vehicles_density": config["experiment"]["vehicles_density"],
        # "duration": config["experiment"]["duration"],
        # "vehicle_speeds": config["experiment"]["vehicle_speeds"],
        # "road_length": config["experiment"]["road_length"],
        # "duration": config["experiment"]["duration"],
        #     "vehicle_type_distribution": {
        #         "sedan": 0.4,
        #         "truck": 0.5,
        #         "semi": 0.1,
        #     },
        #     "render": True,
        #     "high_speed_reward": 0.5,
    })
    environment_config.update(env.config)

    # Make agent
    agent_config = {
        "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
        "env_preprocessors": [{"method": "simplify"}],
        "display_tree": False,
        "budget": config["agent"]["budget"],
        "gamma": config["agent"]["gamma"],
    }
    agent = agent_factory(env, agent_config)

    for seed in tqdm(range(start_seed, end_seed + 1)):
        print(f"seed:{seed}")
        terminated, truncated = False, False
        obs = env.reset(seed=seed)
        states_list = []
        vehicles_counts_list = []
        avg_velocities = []
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
            states_list.append(obs_state)
            vehicles_counts_list.append(len(env.road.vehicles))
            avg_velocities.append(np.mean(
                [v.speed for v in env.road.vehicles]))
            if config["experiment"]["render"]:
                env.render()
        state_record[seed] = states_list
        vehicles_count_record[seed] = vehicles_counts_list
        lane_changes_record[seed] = sum(
            [v.lanes_changed_count for v in env.road.vehicles])
        ego_collisions_record[seed] = env.vehicle.crashed
        avg_velocity[seed] = np.mean(avg_velocities)
    env.close()

# Enable JSON serialization.
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 0

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path

if __name__ == "__main__":
    road_lengths =      [250, 250, 250]
    vehicle_densities = [1.3, 1.3, 1.3]
    lane_counts =       [  2,   3, 4]
    vehicle_counts =    [  8,  12, 16]
    num_sims = 10_000
    for r_len, v_dens, lanes, cars in zip(road_lengths,vehicle_densities,lane_counts, vehicle_counts):
        # Create log directory.
        experiment_dir = op.join(local_highway_env, "scripts", "sensor_scheduling_experiments")
        log_dir = uniquify(op.join(experiment_dir, "experiment"))
        os.makedirs(log_dir, exist_ok=False)
        raw_data_file = op.join(log_dir, "raw_data.json")
        discrete_markov_chain_file = op.join(log_dir, "discrete_markov_chain")
        config_file = op.join(log_dir, "configuration.json")

        # Define data collection configuration.
        first_seed = 1_000
        last_seed = first_seed + num_sims
        config = {
            "experiment": {
                "first_seed": first_seed,
                "last_seed": last_seed,
                "lanes_count": lanes,
                "vehicles_count": cars,
                "vehicles_density": v_dens,
                "road_length": r_len,
                "duration": 30,
                "vehicle_type_distribution": {
                    "sedan": 0.4,
                    "truck": 0.5,
                    "semi": 0.1,
                },
                "render": False,
                "vehicle_speeds": np.linspace(5, 35, 5, endpoint=True),
                # With too low of a high_speed_reward, the agent will stop for a
                # few steps collect maximum rewards and move on. This will cause
                # lots of traffic and collisions.
                "high_speed_reward": 1.0,
            },
            "agent": {
                "budget": 50,
                "gamma": 0.9,
            }
        }
        num_processes = 11
        new_last_seed = ((last_seed - first_seed) // num_processes) \
            * num_processes + first_seed
        step = int((new_last_seed - first_seed) / num_processes)

        # Get ranges for each process.
        # TODO: Can just cycle start_seeds to the left by 1 index and replace
        # the last value to get end_seeds.
        start_seeds = np.arange(first_seed, new_last_seed, step)
        end_seeds = np.arange(first_seed + step, new_last_seed + step, step)
        assert len(start_seeds) == len(end_seeds) == num_processes
        raw_data = {
            "state_record": {}, "vehicles_count_record": {},
            "collisions_count": {}, "steps_till_1st_collision": {}, 
            "environment_config": {}, "lane_changes_record": {}, 
            "ego_collisions_record": {}, "average_velocity": {},
        }
        with multiprocessing.Manager() as manager:
            # Create shared dictionary.
            state_record = manager.dict()
            vehicles_count_record = manager.dict()
            environment_config = manager.dict()
            lane_changes_record = manager.dict()
            ego_collisions_record = manager.dict()
            avg_velocity = manager.dict()
            # Create processes.
            processes = [
                multiprocessing.Process(
                    target=experiment,
                    args=(start, end, config, state_record, ego_collisions_record, 
                        vehicles_count_record, environment_config, 
                        lane_changes_record, avg_velocity))
                for start, end in zip(start_seeds, end_seeds)]
            # Start all processes.
            for process in processes:
                process.start()
            # Wait for all processes to complete.
            for process in processes:
                process.join()

            # Store data from all processes.
            raw_data["state_record"].update(state_record)
            raw_data["vehicles_count_record"].update(vehicles_count_record)
            raw_data["environment_config"].update(environment_config)
            raw_data["lane_changes_record"].update(lane_changes_record)
            raw_data["ego_collisions_record"].update(ego_collisions_record)
            raw_data["average_velocity"].update(avg_velocity)
        # Extract experimental data from the record.
        for seed, vehicles_count_record in raw_data['vehicles_count_record'].items():
            vehicles, counts = np.unique(vehicles_count_record, return_counts=True)
            i_max = np.argmax(vehicles)
            print(f"{seed}: {max(vehicles) - min(vehicles)} collisions")
            print(f"{seed}: {counts[i_max]}s until collision")
            raw_data["collisions_count"][seed] = max(vehicles) - min(vehicles)
            raw_data["steps_till_1st_collision"][seed] = counts[i_max]
        # Build discrete_markov_chain for quick use in analysis.
        mc = discrete_markov_chain(
            raw_data=raw_data["state_record"], num_states=2**16)
        
        # Save markov chain.
        mc.save_object(discrete_markov_chain_file)

        # Save configuration and results.
        num_collisions, collision_freq = np.unique(
            list(raw_data["collisions_count"].values()), return_counts=True)
        time_till_collisions, time_till_freq = np.unique(
            list(raw_data["steps_till_1st_collision"].values()), return_counts=True)
        ego_vehicle_collisions = sum(raw_data["ego_collisions_record"].values())
        print(raw_data["environment_config"])
        config["environment"] = {**raw_data["environment_config"]}
        config["results"] = {
                "num_collisions": num_collisions,
                "collision_freq": collision_freq,
                "time_till_collisions": time_till_collisions,
                "time_till_freq": time_till_freq,
                "ego_vehicle_collisions": ego_vehicle_collisions,
                "lane_changes": sum(raw_data["lane_changes_record"].values()),
                "average_velocity": np.mean(list(raw_data["average_velocity"].values())),
        }
        with open(config_file, "w") as f:
            f.write(json.dumps(config, sort_keys=True, indent=4, cls=NpEncoder))

        with open(raw_data_file, "w") as f:
            f.write(json.dumps(raw_data, sort_keys=True, cls=NpEncoder))
        #print(raw_data)
