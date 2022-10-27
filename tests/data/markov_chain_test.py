from scipy import sparse
import os.path as op
import sys
local_highway_env = op.join(op.dirname(op.realpath(__file__)), "../..",)
sys.path.insert(1, local_highway_env)
import highway_env  # noqa
from highway_env.data.markov_chain import markov_chain  # noqa

num_states = 5
mc = markov_chain(num_states)

irreducible = [
    [0,  1/4,   0, 3/4],
    [1/2,  0, 1/3, 1/6],
    [0,    0,   0,   1],
    [0,  1/2, 1/4, 1/4]
]
irreducible = sparse.csr_matrix(irreducible)

n_components = sparse.csgraph.connected_components(
    irreducible, directed=True, connection='strong', return_labels=False)
print(f"irreducible connected components: {n_components}")

reducible = [
    [0,  1/4,   0, 3/4],
    [1/2,  0, 1/3, 1/6],
    [0,    0,   1,   0],
    [0,  1/2, 1/4, 1/4]
]
reducible = sparse.csr_matrix(reducible)

n_components = sparse.csgraph.connected_components(
    reducible, directed=True, connection='strong', return_labels=False)
print(f"reducible connected components: {n_components}")
