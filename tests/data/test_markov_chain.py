import os
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
    # Able to instantiate with no data.
    mc = discrete_markov_chain(raw_data={0: []}, num_states=3)

    # Able to instantiate with data of unequal length.
    data = {
        0: [0, 0, 0],
        1: [1, 1, 1, 1, 1],
        2: [2, 2, 2, 2],
        3: [0, 0],
    }
    mc = discrete_markov_chain(raw_data=data, num_states=3)
    transition_matrix = mc.transition_matrix
    assert isinstance(transition_matrix, sparse.spmatrix)
    expected_transition_matrix = np.eye(3)
    assert np.linalg.norm(transition_matrix - expected_transition_matrix) == 0.

    # Able to handle case when last state is the only observation of that state
    # and when first state is the only observation of that state.
    data = {
        0: [3, 1, 0, 1],
        1: [0, 1, 0, 2],  # from 2 assume observations to 0, 3
    }
    mc = discrete_markov_chain(raw_data=data, num_states=4)
    transition_matrix = mc.transition_matrix
    expected_transition_matrix = np.array([
        [0, 2/3, 1/3, 0],
        [1, 0, 0, 0],
        [1/2, 0, 0, 1/2],
        [0, 1, 0, 0],
    ])
    assert np.linalg.norm(transition_matrix - expected_transition_matrix) == 0.

    # Able to handle probabilities between [0,1].
    data = {0: [0, 1, 1, 0, 1]}
    mc = discrete_markov_chain(raw_data=data, num_states=2)
    transition_matrix = mc.transition_matrix
    expected_transition_matrix = np.array([
        [0, 1],
        [0.5, 0.5],
    ])
    assert np.linalg.norm(transition_matrix - expected_transition_matrix) == 0.


def test_save_and_load_data():
    data = {
        0: [0, 2, 2, 2],
        1: [0, 2, 2, 2],
    }
    mc = discrete_markov_chain(raw_data=data, num_states=3)
    filename = os.path.join(".", "test_markov_chain")
    mc.save_object(filename)
    mc2 = discrete_markov_chain(raw_data={0: []}, num_states=3)
    mc2.load_object(filename)
    assert list(mc2.transition_data[0]) == data[0]
    expected_transition_matrix = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 1]
    ])
    assert np.linalg.norm(mc2.transition_matrix -
                          expected_transition_matrix) == 0.
    os.remove(filename)


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
    matrix = np.array([
        [0.7, 0.2, 0.1],
        [0.4, 0.6, 0.0],
        [0.0, 1.0, 0.0]
    ])
    mc = discrete_markov_chain(transition_matrix=matrix)
    filename = os.path.join(".", "test_markov_chain")
    mc.save_object(filename)
    mc2 = discrete_markov_chain(transition_matrix=np.array([[1.0]]))
    mc2.load_object(filename + ".npz")
    assert np.linalg.norm(mc2.transition_matrix -
                          matrix) == 0.
    os.remove(filename + ".npz")


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
    mc = discrete_markov_chain(raw_data={0: data}, num_states=1)
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


def test_absolute_discounting():
    # Test on dense distribution.
    eps = 1e-4
    a = sparse.lil_matrix(np.array([[0.1, 0.5, 0.4]]))
    mc = discrete_markov_chain(raw_data={0: []}, num_states=1)
    out = mc.absolute_discount(a, eps)
    assert np.linalg.norm(a - out) < 1e-10
    # Test on sparse distribution.
    dist_a = sparse.lil_matrix(np.array([0.5, 0.0, 0.4, 0.0, 0.2]))
    out = mc.absolute_discount(dist_a, eps)
    expected_out = np.array(
        [0.5 - eps / 3, eps / 2, 0.4 - eps / 3, eps / 2, 0.2 - eps / 3])
    assert np.linalg.norm(out - expected_out) < 1e-10
    assert np.linalg.norm(out) - 1. < 1e-10


def test_compare():
    # Check basic computation of kl divergence.
    A = np.eye(3)
    B = np.eye(3)
    A_mc = discrete_markov_chain(transition_matrix=A)
    B_mc = discrete_markov_chain(transition_matrix=B)
    B_AoU, mean, std = A_mc.compare(B_mc)
    assert B_AoU == 0.
    assert mean == 0.
    assert std == 0.

    # Check handling of nonzero rows and computation of absolute smoothing.
    A = np.array([
        [0.5, 0.2, 0.3, 0.0],
        [0.0, 0.8, 0.2, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    B = np.array([
        [0.5, 0.3, 0.0, 0.2],
        [0.0, 0.8, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    A_mc = discrete_markov_chain(transition_matrix=A)
    B_mc = discrete_markov_chain(transition_matrix=B)
    B_Aou, mean, std = A_mc.compare(B_mc)
    assert B_Aou == 0.25
    eps = 1e-4
    A_expected_0 = np.array([
        [A[0, 0] - eps / 3, A[0, 1] - eps / 3, A[0, 2] - eps / 3, eps],
    ])
    A_expected_1 = np.array([
        [A[1, 1] - eps / 2, A[1, 2] - eps / 2, eps]
    ])
    assert np.sum(A_expected_0) == pytest.approx(1)
    assert np.sum(A_expected_1) == pytest.approx(1)
    B_expected_0 = np.array([
        [B[0, 0] - eps / 3, B[0, 1] - eps / 3, eps, B[0, 3] - eps / 3],
    ])
    B_expected_1 = np.array([
        [B[1, 1] - eps / 2, eps, B[1, 3] - eps / 2],
    ])
    assert np.sum(B_expected_0) == pytest.approx(1)
    assert np.sum(B_expected_1) == pytest.approx(1)
    kl_1 = np.sum(A_expected_0 * np.log2(A_expected_0 / B_expected_0))
    kl_2 = np.sum(A_expected_1 * np.log2(A_expected_1 / B_expected_1))
    assert kl_1 > 0
    assert kl_2 > 0
    assert mean == np.mean([kl_1, kl_2])
    assert std == np.std([kl_1, kl_2])


def test_add():
    data1 = {0: [0, 0, 0, 0, 0],
             1: [1, 2, 1, 2, 1, 2]}
    num_states = 3
    data2 = {2: [0, 0, 0, 0],
             3: [2, 2, 2, 2]}
    mc1 = discrete_markov_chain(raw_data=data1, num_states=num_states)
    mc2 = discrete_markov_chain(raw_data=data2, num_states=num_states)
    data3 = {4: [0, 1, 0, 1, 1, 1],
             5: [1, 2, 0, 1, 2, 0]}
    mc3 = discrete_markov_chain(raw_data=data3, num_states=num_states)
    mc4 = mc1 + mc2 + mc3
    assert mc4.transition_data == list({**data1, **data2, **data3}.values())


def test_dist():
    mc = discrete_markov_chain(raw_data={0: []}, num_states=1)
    assert mc._dist(4, 0) == 1
    assert mc._dist(3, 0) == 2
