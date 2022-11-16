"""
Script to analyze the markov chains collected in data_collection.py
"""
import numpy as np
import os.path as op
from scipy import sparse
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


def sample_correlation_coeff(sparse_matrix):
    """Apply's Pearson Sample Coefficient to matrix to compute correlation
    of the row and columns. A correlation of 1 means the matrix is perfectly
    diagonal. Close to 1 means dominantly diagonal. 
    
    See:
    https://math.stackexchange.com/questions/1392491
    accessed: 11/11/2022
    """
    d = sparse_matrix.shape[0]
    r = np.array(range(d))
    r2 = r**2
    j = np.ones(d)
    n = sparse_matrix.sum()
    sigma_x = r @ sparse_matrix @ j.T
    sigma_y = j @ sparse_matrix @ r.T
    sigma_x2 = r2 @ sparse_matrix @ j.T
    sigma_y2 = j @ sparse_matrix @ r2.T
    sigma_xy = r @ sparse_matrix @ r.T
    return (n*sigma_xy - sigma_x*sigma_y) / np.sqrt(n*sigma_x2 - sigma_x**2) \
        / np.sqrt(n*sigma_y2 - sigma_y**2)


def analyze_matrix(transition_matrix):
    # Compute the proportion of absorbing states.
    absorbing_states = sum(sparse.find(transition_matrix)[2] == 1)
    absorbing_proportion = absorbing_states / transition_matrix.getnnz()
    # Compute the Pearson correlation. 
    corrcoeff = sample_correlation_coeff(transition_matrix)
    return (absorbing_proportion, corrcoeff)
    

def compute_num_collisions(raw_data: dict) -> int:
    data = list(raw_data.values())
    num_collisions = 0
    for i in range(len(data)):
        num_collisions += 1 if len(data[i]) < 30 else 0
    flat_data = []
    for d in data:
        flat_data += d
    states, counts = np.unique(np.array(flat_data),return_counts=True)
    if 0 in states:
        print(counts[0])
    return num_collisions

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

# print(len(set(mc_2_lane.transition_matrix.todok().nonzero()[0])))
mc_2_T, mc_2_map = mc_2_lane.get_irreducible_matrix()
print(f"2 lane (absorbing, diagonality)={analyze_matrix(mc_2_T)}")
mc_2_irreducible = discrete_markov_chain(transition_matrix=mc_2_T)
print(f"2 lane irreducible? {mc_2_irreducible.is_irreducible()}")
print(f"2 lane entropy rate: {mc_2_irreducible.entropy_rate()}")
print(f"Num of collisions: {compute_num_collisions(raw_data)}")

mc_4_T, mc_4_map = mc_4_lane.get_irreducible_matrix()
print(f"4 lane (absorbing, diagonality)={analyze_matrix(mc_4_T)}")
mc_4_irreducible = discrete_markov_chain(transition_matrix=mc_4_T)
print(f"4 lane irreducible? {mc_4_irreducible.is_irreducible()}")
print(f"4 lane entropy rate: {mc_4_irreducible.entropy_rate()}")
print(f"Num of collisions: {compute_num_collisions(mc_4_lane.raw_data)}")
