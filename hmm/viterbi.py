import numpy as np

__author__ = 'eranroz'
"""
Dynamic programming algorithm for decoding the states.
Implementation according to Durbin, Biological sequence analysis [p. 57]
"""


def viterbi(symbol_seq, model):
    """
    Find the most probable path through the model

    @param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
    @param model: an HMM model to calculate on the given symbol sequence
    """
    n_states = model.num_states()
    unique_values = set(symbol_seq)
    emission_seq = np.zeros((len(symbol_seq), n_states - 1))
    for v in unique_values:
        emission_seq[symbol_seq == v, :] = np.log(model.get_emission()[1:, v])

    ptr_mat = np.zeros((len(symbol_seq), n_states-1))
    l_state_trans_mat_T = model.get_state_transition().T

    emission_iterator = iter(emission_seq)
    ptr_iterator = iter(ptr_mat)
    #intial condition is begin state
    prev = next(emission_iterator) + 1+np.log(l_state_trans_mat_T[1:, 0])
    next(ptr_iterator)[...] = np.argmax(prev)
    end_state = 0  # termination step
    end_transition = np.log(l_state_trans_mat_T[end_state, 1:])
    l_state_trans_mat_T = np.log(l_state_trans_mat_T[1:, 1:])
    #recursion step
    for emission_symbol in emission_iterator:
        p_state_transition = prev+l_state_trans_mat_T
        max_k = np.max(p_state_transition, 1)
        next(ptr_iterator)[...] = np.argmax(p_state_transition, 1)
        prev = emission_symbol + max_k

    p_state_transition = prev + end_transition
    last_mat = np.max(p_state_transition)
    #traceback step and without begin state
    most_probable_path = np.zeros(len(symbol_seq), int)
    most_probable_path[-1] = np.argmax(last_mat)

    for i in np.arange(len(symbol_seq)-1, -1, -1):
        most_probable_path[i - 1] = ptr_mat[i, most_probable_path[i]]

    return most_probable_path