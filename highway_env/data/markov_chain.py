from scipy import sparse
import pickle


class markov_chain(object):
    def __init__(self, num_states):
        self._transition_numerator = {}
        self._transition_denominator = {}
        self.num_states = num_states

    def update_transition_matrix(self, prev_state, current_state):
        # If prev_state has been visited in the past increase denominator.
        if prev_state in self._transition_numerator.keys():
            self._transition_denominator[prev_state] += 1
            # If prev_state -> current_state transition has occurred previously,
            # increase counter.
            if current_state in self._transition_numerator[prev_state].keys():
                self._transition_numerator[prev_state][current_state] += 1
            # If transition has not occurred, begin counter at 1.
            else:
                self._transition_numerator[prev_state][current_state] = 1
        # If prev_state has never been visited, initialize numerator with
        # transition counter at 1 and denominator counter at 1.
        else:
            self._transition_numerator[prev_state] = {current_state: 1}
            self._transition_denominator[prev_state] = 1
        # Verify counting correctly.
        assert sum(self._transition_numerator[prev_state].values()) \
            == self._transition_denominator[prev_state]

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

    def get_transition_matrix(self):
        transition_matrix = sparse.lil_array(
            (self.num_states, self.num_states))
        for prev_state in self._transition_numerator.keys():
            for current_state in self._transition_numerator[prev_state].keys():
                transition_matrix[prev_state, current_state] = \
                    self.evaluate_transition_matrix(prev_state, current_state)
        return transition_matrix

    def get_stationary_matrix(self):
        transition_matrix = self.get_stationary_matrix().tocsr()

    def is_irreducible(self, transition_matrix=None):
        if not transition_matrix:
            transition_matrix = self.get_stationary_matrix()
        n_components = sparse.csgraph.connected_components(
            transition_matrix, directed=True, connection='strong',
            return_labels=False)
        return n_components == 1

    def get_irreducible_matrix(self):
        transition_matrix = self.get_stationary_matrix()
        assert self.is_irreducible(transition_matrix)
        return transition_matrix

    def save_object(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
