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
                 raw_data: Optional[dict] = None,
                 num_states: Optional[int] = None,
                 transition_matrix: Union[sparse.spmatrix, np.ndarray] = None):
        # Enforce mutually exclusive instantiation methods.
        assert raw_data is not None or transition_matrix is not None, \
            "Must instantiate with either |transition_data| or |num_states|."

        # If using transition data to create Markov chain.
        if raw_data is not None:
            assert num_states is not None, \
                "|num_states| must be provided when building from data."
            self.num_states = num_states
            self.transition_data = list(raw_data.values())
            self.raw_data = raw_data
        # If using transition matrix to create Markov chain.
        else:
            self.transition_matrix = transition_matrix

    def __add__(self, markov_chain_b):
        assert isinstance(markov_chain_b, discrete_markov_chain)
        assert self.num_states == markov_chain_b.num_states
        # If the same seed was used to generate data in chain a and chain b,
        # the data associated with that seed is overwritten with chain b's data.
        combined_raw_data = {**self.raw_data, **markov_chain_b.raw_data}
        return discrete_markov_chain(raw_data=combined_raw_data,
                                     num_states=self.num_states)

    @property
    def transition_data(self) -> List:
        return self._transition_data

    @transition_data.setter
    def transition_data(self, transition_data: Union[List, np.ndarray]):
        # Store transition_data as list of lists to support variable lengths of
        # separate data entries.
        if isinstance(transition_data, np.ndarray):
            transition_data = list(transition_data)
        # If not a list of lists, convert to a list of lists.
        try:
            if not isinstance(transition_data[0], list):
                transition_data = [transition_data]
        # Handle empty 1 depth lists.
        except IndexError:
            transition_data = [transition_data]
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

    def absolute_discount(self, dist_a, eps):
        """
        Fills dist_a to be dense but maintaining the sum at 1.
        """
        a = dist_a.todense()
        a_nonzero = a.nonzero()[1]
        a_zero = np.nonzero(a == 0.)[1]
        if len(a_zero) > 0:
            a[0, a_nonzero] -= eps / len(a_nonzero)
            a[0, a_zero] = eps / len(a_zero)
        return a

    def get_outlying_states(self, markov_chain_b):
        """Returns the states and number of states in chain a that don't 
        belong to chain b and the states of b that don't belong to chain a."""
        chain_a_matrix = self.transition_matrix.todok()
        chain_b_matrix = markov_chain_b.transition_matrix.todok()
        a_nonzero_rows = set(np.unique(chain_a_matrix.nonzero()[0]))
        b_nonzero_rows = set(np.unique(chain_b_matrix.nonzero()[0]))
        # First analyze the sets of states observed in both markov chains.
        a_union_b = a_nonzero_rows.union(b_nonzero_rows)
        a_intersect_b = a_nonzero_rows.intersection(b_nonzero_rows)
        states_not_in_a_or_b = a_union_b - a_intersect_b
        state_mismatch_count = len(states_not_in_a_or_b)
        return (states_not_in_a_or_b, state_mismatch_count)

    def compare(self, markov_chain_b,
                eps: Optional[float] = 1e-4) -> Tuple:
        """
        Compare to markov_chain_b by computing mean of the KL divergence with
        absolute discounting of each row of the two markov chains. Rows are
        skipped if they are empty in either markov chain.
        """
        # Analyze the sets of chain a and chain b.
        chain_a_matrix = self.transition_matrix.todok()
        chain_b_matrix = markov_chain_b.transition_matrix.todok()
        a_nonzero_rows = set(chain_a_matrix.nonzero()[0])
        b_nonzero_rows = set(chain_b_matrix.nonzero()[0])
        a_union_b = a_nonzero_rows.union(b_nonzero_rows)
        a_intersect_b = a_nonzero_rows.intersection(b_nonzero_rows)
        b_states_not_in_a = b_nonzero_rows - a_nonzero_rows
        # Compute row-wise kl divergence with absolute smoothing.
        kl_divs = []
        for row in a_intersect_b:
            a_row = chain_a_matrix.getrow(row).tolil()
            b_row = chain_b_matrix.getrow(row).tolil()
            # Remove where both a and b are zero.
            a_nonzero = set(a_row.nonzero()[1])
            b_nonzero = set(b_row.nonzero()[1])
            a_col_union_b_col = list(a_nonzero.union(b_nonzero))
            a_row = self.absolute_discount(a_row[0, a_col_union_b_col], eps)
            b_row = self.absolute_discount(b_row[0, a_col_union_b_col], eps)
            kl_divs.append(np.sum(np.multiply(a_row, np.log2(a_row / b_row))))
        return (len(b_states_not_in_a) / len(a_union_b), np.mean(kl_divs), np.std(kl_divs))

    def entropy_rate(self) -> float:
        stationary_distribution = self.get_stationary_distribution()
        entropy_rate = 0
        P = self.transition_matrix.tocoo()
        for i, P_ij in zip(P.row, P.data):
            entropy_rate += stationary_distribution[i] * P_ij * np.log2(P_ij)
        entropy_rate *= -1
        return entropy_rate

    def simplify_matrix(self) -> np.ndarray:
        """Returns a matrix with unobserved states removed."""
        nonzero_rows, nonzero_cols = self.transition_matrix.nonzero()
        observed_rows = set(nonzero_rows)
        observed_cols = set(nonzero_cols)
        return (observed_rows, observed_cols)

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
                pickle.dump((self.raw_data, self.num_states), file,
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
                raw_data, self.num_states = pickle.load(file)
                self.transition_data = list(raw_data.values())
                self.raw_data = raw_data

    def _get_transition_matrix_from_data(self) -> sparse.spmatrix:
        # If data is 1D then assume it is from a single experiment. If 2D then
        # each row is a separate experiment.
        # Build frequency matrix by counting all the transitions in data.
        frequency_matrix = sparse.lil_matrix(
            (self.num_states, self.num_states), dtype=np.uint16)
        for row in self.transition_data:
            # Build 1st order markov chain transition matrix. Skip the first
            # state in the data to count transitions.
            for j, current_state in enumerate(row[1:]):
                past_state = row[j]
                # Assume transition data is encoded as integers.
                frequency_matrix[past_state, current_state] += 1
        rows_set = self._remove_unobserved_states(frequency_matrix)
        # Normalize frequency matrix to get transition_matrix.
        transition_matrix = sparse.lil_matrix(
            (self.num_states, self.num_states), dtype=np.float64)
        for row in rows_set:
            denominator = np.sum(frequency_matrix[row, :])
            transition_matrix[row, :] = \
                frequency_matrix[row, :].astype(np.float64) / denominator
        return transition_matrix.asformat('csr')

    def _remove_unobserved_states(self, frequency_matrix):
        """
        It is possible that the data ends on a state that has never been seen
        before. This will cause the column corresponding to that state to be
        nonzero, but the row will be zero. When simulating this transition
        matrix, this can cause problems. We can either make this an absorbing
        state, remove the observation, or make some assumption about the next
        state. For now, we assume the next state is a distance 1 from the
        current state.

        It is possible that the data starts on a state that is never seen
        again. This will cause the column corresponding to that state to be
        zero, but the row will be nonzero. This is not a problem when
        simulating this transition matrix so this is not dealt with.
        """
        def find_closest_states(state, observed_states, d=1):
            close_states = [s for s in observed_states
                            if self._dist(state, s) == d]
            if len(close_states) > 0:
                return close_states
            else:
                close_states = find_closest_states(state,
                                                   observed_states,
                                                   d + 1)
                return close_states

        # Find the states in the columns that are not in the rows.
        nonzero_rows, nonzero_cols = frequency_matrix.nonzero()
        rows_set, cols_set = set(nonzero_rows), set(nonzero_cols)
        outlier_states = cols_set - rows_set
        for outlier_state in outlier_states:
            # Find the previously observed states that are the closest.
            close_states = find_closest_states(outlier_state, rows_set)
            frequency_matrix[outlier_state,
                             close_states] += 1 * np.ones(len(close_states))
        # It is possible that removing the last observation from an experiment
        # will expose a new last state that has never been visited before.
        nonzero_rows, nonzero_cols = frequency_matrix.nonzero()
        rows_set, cols_set = set(nonzero_rows), set(nonzero_cols)
        assert len(cols_set - rows_set) == 0
        return rows_set

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
