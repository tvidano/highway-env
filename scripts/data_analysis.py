"""
Script to analyze the markov chains collected in data_collection.py
"""
import numpy as np
import os.path as op
import sys

# use local version of highway_env, before simulation you should be told this
# is a local version.
local_highway_env = op.join(op.dirname(op.realpath(__file__)), "..",)
sys.path.insert(1, local_highway_env)
import highway_env  # noqa
from highway_env.data.markov_chain import discrete_markov_chain  # noqa


def compare_across_seeds(raw_data, resolution):
    seeds = list(raw_data.keys())
    new_data = {k: v for k, v in raw_data.items() if k == min(seeds)}
    max_seed = min(seeds)
    cumulative_mc = discrete_markov_chain(num_states=2**16, raw_data=new_data)
    for seed in range(min(seeds) + resolution, max(seeds) + 1, resolution):
        new_data = {k: v for k, v in raw_data.items() if max_seed <= k < seed}
        new_mc = discrete_markov_chain(num_states=2**16, raw_data=new_data)
        cumulative_min_seed = min(cumulative_mc.raw_data.keys())
        cumulative_max_seed = max(cumulative_mc.raw_data.keys())
        new_min_seed = min(new_mc.raw_data.keys())
        new_max_seed = max(new_mc.raw_data.keys())
        print(
            f"{cumulative_min_seed}-{cumulative_max_seed}->"\
            f"{new_min_seed}-{new_max_seed}: {cumulative_mc.compare(new_mc)}")
        cumulative_mc = cumulative_mc + new_mc
        max_seed = seed


mc1 = discrete_markov_chain(num_states=2**16, raw_data={1000: []})
mc1.load_object("2_lane_low_density_low_cars_1000_4999")
mc2 = discrete_markov_chain(num_states=2**16, raw_data={5000: []})
mc2.load_object("2_lane_low_density_low_cars_5000_10999")
raw_data = (mc1 + mc2).raw_data
mc_2_lane = discrete_markov_chain(num_states=2**16, raw_data=raw_data)
mc_4_lane = discrete_markov_chain(num_states=2**16, raw_data={1000: []})
mc_4_lane.load_object("4_lane_low_density_low_cars_1000_10999")

# print("Comparing 2 lane highway experiment.")
# compare_across_seeds(mc_2_lane.raw_data, 1000)

# print("Comparing 4 lane highway experiment")
# compare_across_seeds(mc_4_lane.raw_data, 1000)

# print(mc_2_lane.compare(mc_4_lane))
# print(mc_4_lane.compare(mc_2_lane))

print(len(set(mc_2_lane.transition_matrix.todok().nonzero()[0])))
print(len(set(mc_4_lane.transition_matrix.todok().nonzero()[0])))
