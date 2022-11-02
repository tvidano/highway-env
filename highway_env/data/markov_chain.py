from optparse import Option
from scipy import sparse
from scipy.sparse import csgraph
import pickle
from typing import Dict, Text, List, Tuple, Optional, Callable, Union
import numpy as np
import os


class discrete_markov_chain(object):
    """
    A discrete markov chain that can be constructed from experimental data or
    a transition matrix.
    """

    def __init__(self, *,
                 transition_data: Optional[np.ndarray] = None,
                 num_states: Optional[int] = None,
                 transition_matrix: Union[sparse.spmatrix, np.ndarray] = None):
        # Enforce mutually exclusive instantiation methods.
        assert transition_data is not None or transition_matrix is not None, \
            "Must instantiate with either |transition_data| or |num_states|."

        # If using transition data to create Markov chain.
        if transition_data is not None:
            assert num_states is not None, \
                "|num_states| must be provided when building from data."
            self.num_states = num_states
            self.transition_data = transition_data
        # If using transition matrix to create Markov chain.
        else:
            self.transition_matrix = transition_matrix

    @property
    def transition_data(self) -> List:
        return self._transition_data

    @transition_data.setter
    def transition_data(self, transition_data: List):
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
        if isinstance(transition_matrix, np.ndarray):
            transition_matrix = sparse.csr_matrix(transition_matrix)
            transition_matrix.eliminate_zeros()
            self._transition_matrix = transition_matrix
        elif isinstance(transition_matrix, sparse.spmatrix):
            transition_matrix.eliminate_zeros()
            self._transition_matrix = transition_matrix.asformat('csr')

    def get_stationary_distribution(
            self, transition_matrix: Optional[sparse.spmatrix] = None,
            tol: Optional[float] = 1e-12, method: Optional[str] = 'eigen'):
        """
        Compute the stationary distributions (pi in pi = pi @ P) for the sparse
        transition matrix, P. If P is reducible solves for the stationary
        distributions of the smallest strongly connected component.

        https://stackoverflow.com/questions/21308848/
        accessed: 10-27-2022
        """
        method_list = ['eigen', 'Krylov', 'power']
        assert method in method_list, \
            f"{method} not supported, please choose from {method_list}"

        if transition_matrix is None:
            P = self.transition_matrix
            n = self.num_states
        else:
            P = transition_matrix
            n = transition_matrix.shape[0]

        # Separate connected components.
        n_components, labels = csgraph.connected_components(
            P, directed=True, connection='strong')
        if n_components > 1:
            # Remove decaying components from labels.
            index_sets = []
            for i in range(n_components):
                indices = np.flatnonzero(labels == i)
                other_indices = np.flatnonzero(labels != i)

                Px = P[indices, :][:, other_indices]
                if Px.max() == 0:
                    index_sets.append(indices)
            n_components = len(index_sets)

            # Select the smallest connected component.
            sizes = [indices.size for indices in index_sets]
            min_i = np.argmin(sizes)
            indices = index_sets[min_i]

            # Solve stationary state for it.
            p = np.zeros(n)
            if indices.size == 1:
                p[indices] = 1
            else:
                p[indices] = self.get_stationary_distribution(
                    P[indices, :][:, indices], tol=tol, method=method)
            return p

        # Solve for stationary distribution of irreducible matrix.
        else:
            if P.shape == (1, 1):
                return np.array([1.0])

            P_minus_I = P - sparse.eye(n)
            lhs = sparse.vstack([np.ones(n), P_minus_I.T[1:, :]])
            # GMRES does not support rhs being sparse.spmatrix for some reason.
            rhs = np.zeros((n,))
            rhs[0] = 1

            if method == "eigen":
                # Assumes solution is unique.
                return sparse.linalg.spsolve(lhs, rhs)
            elif method == "Krylov":
                # Assumes the first solution found by searching the Krylov
                # subspace is the desired solution.
                p, info = sparse.linalg.gmres(lhs, rhs, tol=tol)
                if info != 0:
                    raise RuntimeError("gmres didn't converge.")
                return p
            elif method == "power":
                return NotImplementedError
            else:
                return ValueError(f"{method} unrecognized.")

    def compare(self, markov_chain_b):
        # TODO: determine how to compare two markov chains. The ultimate goal is
        # to evaluate how wrong chain b is at predicting trajectories from
        # chain a.
        return NotImplementedError

    def entropy_rate(self):
        stationary_distribution = self.get_stationary_distribution()
        entropy_rate = 0
        P = self.transition_matrix.tocoo()
        for i, P_ij in zip(P.row, P.data):
            entropy_rate += stationary_distribution[i] * P_ij * np.log2(P_ij)
        entropy_rate *= -1
        return entropy_rate

    def is_irreducible(self, transition_matrix=None):
        if transition_matrix is None:
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

    def save_object(self,
                    filename: str):
        # Determine if Markov chain is defined using a transition matrix or
        # transition data.
        is_defined_transition_data = hasattr(self, "_transition_data")
        if is_defined_transition_data and len(self.transition_data) == 0:
            raise ValueError(
                "Trying to save a markov chain with no transition_data")
        elif is_defined_transition_data:
            _, ext = os.path.splitext(filename)
            assert ext != ".npz", "Cannot save transition data as .npz file."
            with open(filename, "wb") as file:
                pickle.dump((self.transition_data, self.num_states), file,
                            pickle.HIGHEST_PROTOCOL)
        elif not is_defined_transition_data:
            sparse.save_npz(filename, self.transition_matrix)
        else:
            raise NotImplementedError

    def load_object(self, filename):
        _, ext = os.path.splitext(filename)
        if ext == ".npz":
            self.transition_matrix = sparse.load_npz(filename)
        else:
            with open(filename, "rb") as file:
                transition_data, self.num_states = pickle.load(file)
                self.transition_data = transition_data

    def _get_transition_matrix_from_data(self) -> sparse.spmatrix:
        # If data is 1D then assume it is from a single experiment. If 2D then
        # each row is a separate experiment.
        
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

    def _dist(self, current_state: int, next_state: int) -> int:
        """
        Computes Hamming Distance for state encodings. Assumes the state is an
        integer encoding of a binary vector where each item in the vector 
        corresponds to occupancy state of an occupancy grid. 

        Example:
            Occupancy grid has 4 spaces where each state can be occupied = 1, 
            or unoccupied = 0. The state is therefore X1 = {0, 0, 0, 0} for a
            fully unoccupied grid, and X2 = {1, 1, 1, 1} for a fully occupied 
            grid. The distance between X1 and X2 is best represented by the
            Hamming distance. This can be easily computed by the XOR bitwise
            operator in Python. X1 = {0, 0, 0, 0} = 0, X2 = {1, 1, 1, 1} = 16.
            self._dist(0, 16) = 4. Since only 4 bits are required to be flipped
            to change the state from 0 to 16.
        """
        def bit_count(n):
            count = 0
            while n > 0:
                count += 1
                # Eliminate the least significant 1 in binary format of n.
                n = n & (n - 1)
            return count
        
        hamming_distance = current_state ^ next_state
        return bit_count(hamming_distance)