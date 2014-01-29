import numpy as np
from collections import namedtuple

__author__ = 'eranroz'


def forward_backward(symbol_seq, model, model_end_state=False):
    """
    Calculates the probability for the model and each step in it

    @param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
    @param model: an HMM model to calculate on the given symbol sequence
    @param model_end_state: whether to consider end state or not

    Remarks:
    this implementation uses scaling variant to overcome floating points errors.
    """
    n_states = model.num_states()
    emission = model.get_emission()
    state_trans_mat = model.get_state_transition()

    s_j = np.ones(len(symbol_seq))
    forward = np.zeros((len(symbol_seq), n_states - 1), order='F')  # minus the begin state
    backward = np.zeros((len(symbol_seq), n_states - 1), order='F')

    #-----    forward algorithm   -----
    #intial condition is begin state (in Durbin there is another forward - the begin = 1)
    forward[0, :] = state_trans_mat[0, 1:] * emission[1:, symbol_seq[0]]
    s_j[0] = sum(forward[0, :])
    forward[0, :] /= s_j[0]
    prev_forward = forward[0, :]
    #recursion step
    #transform to emission array
    unique_values = set(symbol_seq)
    emission_seq = np.zeros((len(symbol_seq), n_states - 1))
    for v in unique_values:
        emission_seq[symbol_seq == v, :] = emission[1:, v]

    real_transitions = state_trans_mat[1:, 1:]
    t_real_transitions = real_transitions.transpose()
    p_state_transition = np.zeros(n_states - 1)
    summing_arr = np.array([1] * (n_states - 1))
    emission_iterator = iter(emission_seq)
    next(emission_iterator)  # skip the first (instead of condition in for loop
    s_j_iterator = np.nditer(s_j, op_flags=['writeonly'])
    forward_iterator = np.nditer(forward, flags=['external_loop'], op_flags=['writeonly'], order='C')
    next(s_j_iterator)
    next(forward_iterator)
    for sym_emission in emission_iterator:
        #p_state_transition = np.sum(real_transitions * prev_forward[:, None], 0)
        np.dot(t_real_transitions, prev_forward, p_state_transition)
        prev_forward = sym_emission * p_state_transition

        # scaling - see Rabiner p. 16, or Durbin p. 79
        scaling = np.dot(summing_arr, prev_forward)  # dot is actually faster then np.sum(prev_forward)
        next(s_j_iterator)[...] = scaling
        #s_j[i] = scaling
        #forward[i, :] = prev_forward = prev_forward / scaling
        next(forward_iterator)[...] = prev_forward = prev_forward / scaling

    #end transition
    log_p_model = np.sum(np.log(s_j))
    if model_end_state:  # Durbin - with end state
        end_state = 0  # termination step
        end_transition = forward[len(symbol_seq) - 1, :] * state_trans_mat[1:, end_state]
        log_p_model += np.log(sum(end_transition))

    #-----  backward algorithm  -----
    #intial condition is end state
    if model_end_state:
        prev_back = backward[len(symbol_seq) - 1, :] = (state_trans_mat[1:, 0])  # Durbin p.60
    else:
        prev_back = backward[len(symbol_seq) - 1, :] = [1, 1]  # Rabiner p.7 (24)

    backward_iterator = np.nditer(backward[::-1], flags=['external_loop'], op_flags=['writeonly'], order='C')
    next(backward_iterator)
    s_j_iterator = iter(s_j[::-1])

    #recursion step
    for sym_emission in emission_seq[:0:-1]:
        np.dot(prev_back * sym_emission, t_real_transitions, p_state_transition)
        #backward[i - 1, :] = prev_back = p_state_transition / s_j[i]  # same scaling as in the forward
        next(backward_iterator)[...] = prev_back = p_state_transition / next(s_j_iterator)

    bf_result = namedtuple('BFResult', 'model_p state_p forward backward scales')
    return bf_result(log_p_model, backward * forward, forward, backward, s_j)


def forward_backward_log(symbol_seq, model, model_end_state=False):
    """
    Calculates the probability for the model and each step in it

    @param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
    @param model: an HMM model to calculate on the given symbol sequence
    @param model_end_state: whether to consider end state or not

    Remarks:
    this implementation uses log variant to overcome floating points errors.
    """
    interpolation_res = 0.01
    interpolation_res_i = 1.0 / interpolation_res
    interpol_tbl = np.log(1 + np.exp(-np.arange(0, 35, interpolation_res)))
    last_interpol = len(interpol_tbl) - 1

    def interpolate(prev):
        """
        Uses interpolation table to calculate log r=log(p+log(1+exp(q-p))) which is approx equals to
        log r = log (max)+(exp(1+log(x)) [x=min-max from prev].
        see Durbin p. 79
        @param prev: previous probabilities plus the transition
        @return: result of interpolation of log r=log(p+log(1+exp(q-p)))
        """
        maxes = np.max(prev, 1)
        interpolation_i = np.minimum(np.round(-interpolation_res_i * (np.sum(prev, 1) - 2 * maxes)),
                                     last_interpol).astype(int)
        return maxes + interpol_tbl[interpolation_i]

    l_emission = np.log(model.get_emission()[1:, :])
    forward = np.zeros((len(symbol_seq), model.num_states() - 1))
    backward = np.zeros((len(symbol_seq), model.num_states() - 1))

    l_state_transition = np.log(model.get_state_transition()[1:, 1:])
    l_t_state_transition = l_state_transition.transpose()
    #-----    forward algorithm   -----
    #intial condition is begin state (in Durbin there is another forward - the begin = 1)
    prev_forward = forward[0, :] = np.log(model.get_state_transition()[0, 1:]) + l_emission[:, symbol_seq[0]]
    #recursion step
    emission_seq = list(enumerate([l_emission[:, s] for s in symbol_seq]))

    for i, sym_emission in emission_seq:
        if i == 0:
            continue

        from_prev = prev_forward + l_t_state_transition  # each row is different state, each col - CHECKED
        # now the sum approximation
        from_prev = interpolate(from_prev)
        forward[i, :] = prev_forward = sym_emission + from_prev

    # termination step
    if model_end_state:
        end_state = 0
        log_p_model = forward[len(symbol_seq) - 1, :] + np.log(model.get_state_transition()[1:, end_state])
        log_p_model = interpolate(np.array([log_p_model]))
    else:
        log_p_model = interpolate([forward[len(symbol_seq) - 1, :]])

    #-----  backward algorithm  -----
    last_index = len(symbol_seq) - 1
    if model_end_state:
        prev_back = backward[last_index, :] = np.log(model.get_state_transition()[1:, 0])
    else:
        prev_back = backward[last_index, :] = [0, 0]  # Rabiner p.7 (24)

    for i, sym_emission in reversed(emission_seq):
        if i == 0:
            continue
        p_state_transition = interpolate(l_state_transition + (prev_back + sym_emission))
        prev_back = backward[i - 1, :] = p_state_transition

    # posterior probability 3.14 (P.60 Durbin)
    l_posterior = forward + backward - log_p_model
    bf_result = namedtuple('BFResult', 'model_p state_p forward backward')
    return bf_result(log_p_model, l_posterior, forward, backward)