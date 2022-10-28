from tkinter import Entry
import numpy as np
import pytest

from scipy import sparse
import os.path as op
import sys
local_highway_env = op.join(op.dirname(op.realpath(__file__)), "../..",)
sys.path.insert(1, local_highway_env)
import highway_env  # noqa
from highway_env.data.markov_chain import discrete_markov_chain  # noqa


def test_markov_from_data():
    data = [0, 2, 2, 2]
    mc = discrete_markov_chain(transition_data=data, num_states=3)
    transition_matrix = mc.transition_matrix
    assert isinstance(transition_matrix, sparse.spmatrix)
    expected_transition_matrix = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 1]
    ])
    assert np.linalg.norm(transition_matrix - expected_transition_matrix) == 0.


def test_save_and_load_data():
    # TODO: Test saving and loading from data.
    return NotImplementedError


def test_markov_from_matrix():
    matrix = np.array([
        [0.7, 0.2, 0.1],
        [0.4, 0.6, 0.0],
        [0.0, 1.0, 0.0]
    ])
    mc = discrete_markov_chain(transition_matrix=matrix)
    assert isinstance(mc.transition_matrix, sparse.spmatrix)
    assert np.linalg.norm(mc.transition_matrix - matrix) == 0.


def test_save_and_load_matrix():
    # TODO: Test saving and loading from a matrix.
    return NotImplementedError


def test_irreducible():
    irreducible = np.array([
        [0,  1/4,   0, 3/4],
        [1/2,  0, 1/3, 1/6],
        [0,    0,   0,   1],
        [0,  1/2, 1/4, 1/4]
    ])
    mc = discrete_markov_chain(transition_matrix=irreducible)
    assert mc.is_irreducible()


def test_reducible():
    reducible = np.array([
        [0,  1/4,   0, 3/4],
        [1/2,  0, 1/3, 1/6],
        [0,    0,   1,   0],
        [0,  1/2, 1/4, 1/4]
    ])
    mc = discrete_markov_chain(transition_matrix=reducible)
    assert not mc.is_irreducible()


def test_stationary_matrix():
    # Test simple irreducible matrix.
    mc = discrete_markov_chain(transition_matrix=np.array([
        [0.7, 0.2, 0.1],
        [0.4, 0.6, 0.0],
        [0.0, 1.0, 0.0]
    ]))
    assert mc.is_irreducible()
    pi = mc.get_stationary_distribution()
    error = pi @ mc.transition_matrix - pi
    assert np.linalg.norm(error) < 1e-12
    pi = mc.get_stationary_distribution(method='Krylov')
    error = pi @ mc.transition_matrix - pi
    assert np.linalg.norm(error) < 1e-12

    # Test with reducible matrix.
    n = 100_000
    P = sparse.eye(n)
    P += sparse.diags([1, 1], [-1, 1], shape=P.shape)
    # Disconnect several components.
    P = P.tolil()
    P[:1000, 1000:] = 0
    P[1000:, :1000] = 0
    P[10_000:11_000, :10_000] = 0
    P[10_000:11_000, 11_000:] = 0
    P[:10_000, 10_000:11_000] = 0
    P[11_000:, 10_000:11_000] = 0
    # Normalize to create probability matrix.
    P = P.tocsr()
    P = P.multiply(sparse.csr_matrix(1/P.sum(1).A))
    mc = discrete_markov_chain(transition_matrix=P)
    assert not mc.is_irreducible()
    pi = mc.get_stationary_distribution()
    error = pi @ mc.transition_matrix - pi
    assert np.linalg.norm(error) < 1e-12
    pi = mc.get_stationary_distribution(method='Krylov')
    error = pi @ mc.transition_matrix - pi
    assert np.linalg.norm(error) < 1e-11


def test_entropy_rate():
    # Test trivial problem.
    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mc = discrete_markov_chain(transition_data=data, num_states=1)
    assert mc.is_irreducible()
    assert abs(mc.entropy_rate()) < 1e-12

    # Test entropy rate calculation for simple chain.
    P = np.array([
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 1.0, 0.0]
    ])
    expected_entropy_rate = 0.5
    assert expected_entropy_rate > 0
    mc = discrete_markov_chain(transition_matrix=P)
    entropy_rate = mc.entropy_rate()
    assert entropy_rate > 0
    assert abs(entropy_rate - expected_entropy_rate) < 1e-12

# TODO: Add tests for compare().
