from scipy import sparse
import pickle
from typing import Dict, Text, List, Tuple, Optional, Callable, Union
import numpy as np


class discrete_markov_chain(object):
    """
    A discrete markov chain that can be constructed from experimental data or
    a transition matrix.
    """

    def __init__(self, *, transition_data: Optional[List] = None,
                 transition_matrix: Union[sparse.spmatrix, np.ndarray] = None):
        assert transition_data is not None or transition_matrix is not None, \
            "Must instantiate with either |transition_data| or |num_states|."
        if transition_data:
            self.transition_data = transition_data
        else:
            self.transition_data = []
            self.transition_matrix = transition_matrix
        self._transition_numerator = sparse.lil_array(
            (self.num_states, self.num_states))
        self._transition_denominator = sparse.lil_array((self.num_states, 1))

    @property
    def transition_data(self) -> List:
        return self._transition_data

    @transition_data.setter
    def transition_data(self, transition_data: List):
        self.num_states = len(np.unique(transition_data))
        self._transition_data = transition_data
        self.transition_matrix = self._get_transition_matrix_from_data()

    @property
    def transition_matrix(self) -> sparse.spmatrix:
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(
            self,
            transition_matrix: Union[sparse.spmatrix, np.ndarray]):
        dim_1, dim_2 = transition_matrix.shape
        assert dim_1 == dim_2, "transition matrix must be square."
        self.num_states = dim_1
        self._transition_matrix = transition_matrix

    def evaluate_transition_matrix(self, current_state, next_state):
        # If current_state has been visited, check if next_state has also been
        # visited.
        if current_state in self._transition_numerator.keys():
            if next_state in self._transition_numerator[current_state].keys():
                numerator = self._transition_numerator[current_state][next_state]
            # If current_state->next_state transition has never occurred,
            # the probability is 0.
            else:
                numerator = 0.
            denominator = self._transition_denominator[current_state]
        # If current_state has never been visited, the probability is 0.
        else:
            return 0.
        # If current_state->next_state transition has occurred return the
        # probability.
        return numerator / denominator

    def get_stationary_matrix(self):
        return NotImplementedError

    def is_irreducible(self, transition_matrix=None):
        if not transition_matrix:
            transition_matrix = self.transition_matrix
        n_components = sparse.csgraph.connected_components(
            transition_matrix, directed=True, connection='strong',
            return_labels=False)
        return n_components == 1

    def get_irreducible_matrix(self):
        transition_matrix = self.transition_matrix
        assert self.is_irreducible(transition_matrix), \
            "Markov chain has more than 1 strongly connected components."
        return transition_matrix

    def save_object(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.transition_data, file, pickle.HIGHEST_PROTOCOL)

    def load_object(self, filename):
        with open(filename, "rb") as file:
            self.transition_data = pickle.load(file)

    def _get_transition_matrix_from_data(self) -> sparse.lil_matrix:
        # Get all starting states and their frequencies. Do not include the last
        # state because it is not a starting state in a transition.
        states, states_frequency = np.unique(self.transition_data[:-1],
                                             return_counts=True)
        denominator_dict = {
            state: state_frequency for state, state_frequency in
            zip(states, states_frequency)}
        # Get numerator matrix from the transition_data.
        transition_matrix = sparse.lil_matrix(
            (self.num_states, self.num_states), dtype=np.float64)
        # Build 1st order markov chain transition matrix by iterating through
        # all the data. Skip the first state in the data to count transitions.
        for i, current_state in enumerate(self.transition_data[1:]):
            past_state = self.transition_data[i]
            # Assume transition data is encoded as integers.
            transition_matrix[past_state, current_state] += 1 / \
                denominator_dict[past_state]
        return transition_matrix.asformat('csr')
