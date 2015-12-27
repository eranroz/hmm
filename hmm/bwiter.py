# coding=utf-8
__author__ = 'eranroz'
"""
Implementation of Baum-Welch, or re-estimation procedure.
See:
* Durbin p. 64
* Rabiner p. 8 and p. 17
"""
from copy import deepcopy
import time
import numpy as np
from dynamic import HMMModel, fbward

_time_profiling = False


class IteratorCondition():
    """
    Iteration condition class for Bau-Welch with limited number of iterations
    which also outputs the likelihood for each iteration
    """

    def __init__(self, count):
        """
        @param count: number of iterations
        """
        self.i = count
        self.start = False

    def __call__(self, *args, **kwargs):
        if self.start:
            print('\t\t\t====(#: %s) Likelihood:' % self.i, args[0], '=====')
        else:
            self.start = True
        self.i -= 1
        return self.i >= 0


def bw_iter(symbol_seq, initial_model=None, stop_condition=IteratorCondition(3)):
    """
    Estimates model parameters using Baum-Welch algorithm

    @param symbol_seq: observations
    @param initial_model: Initial guess for model parameters
    @type initial_model: C{ContinuousHMM} or C{DiscreteHMM}
    @param stop_condition: Callable like object/function that takes probability as parameter
    @return: HMM model estimation
    """
    new_model = deepcopy(initial_model)
    if isinstance(stop_condition, int):
        stop_condition = IteratorCondition(stop_condition)

    prob = 0
    while stop_condition(prob):
        if _time_profiling:
            bef = time.time()
        bw_output = fbward.forward_backward(symbol_seq, new_model)
        if _time_profiling:
            print(time.time()-bef)
        prob = bw_output.model_p
        new_model.maximize(symbol_seq, bw_output)

    return new_model, prob


def bw_iter_log(symbol_seq, initial_model=None, n_states=0, n_alphabet=0, stop_condition=None):
    """
    Estimates model parameters using Baum-Welch algorithm with log backward forward

    @param stop_condition: stop condition for EM iterations. default is 10
    @param n_alphabet: length of the alphabet
    @param n_states: number of state
    @param symbol_seq: observations
    @param initial_model: Initial guess for model parameters
    @return: HMM model estimation

    @attention it is better to use scaled version as above. implementation is provided as a reference and for comparing
    to scaled version - but it is less tested, and may insert small inaccuracies
    """
    if initial_model is None:
        r_state_transition = np.zeros((n_states, n_states))
        rand_p = np.random.rand(n_states - 1, n_states - 1)
        rand_p /= np.sum(rand_p, 1)
        r_state_transition[1:, 1:] = rand_p

        rand_p = np.random.rand(n_states - 1)  # begin state
        r_state_transition[0, 1:] = rand_p / np.sum(rand_p, 0)
        rand_p = np.random.rand(n_states - 1)  # end state
        r_state_transition[1:, 0] = rand_p / np.sum(rand_p, 0)
        # random emission values, it may not be the best to choose random, see P. 18 in Rabiner
        r_emission = np.random.rand(n_states, n_alphabet)
        r_emission /= np.sum(r_emission, 1)[:, None]
        initial_model = HMMModel.HMMModel(r_state_transition, r_emission)
    else:
        n_states = initial_model.num_states()
        n_alphabet = initial_model.num_alphabet()

    if stop_condition is None:
        stop_condition = IteratorCondition(10)

    new_model = initial_model
    prob = 0
    while stop_condition(prob):
        bw_output = fbward.forward_backward_log(symbol_seq, new_model)

        # find new state transition matrix and emission matrix
        new_state_transition = np.zeros((n_states, n_states))
        l_emission = np.log(new_model.get_emission()[1:, :])
        eb_symbols = np.array([l_emission[:, sym] for sym in symbol_seq])
        eb_symbols += bw_output.backward

        for k in range(1, n_states):
            fbEm = eb_symbols[1:, :] + bw_output.forward[:-1, k-1][:, None]
            new_state_transition[k, 1:] = new_model.state_transition[k, 1:]*np.sum(np.exp(fbEm - bw_output.model_p), 0)#

        new_state_transition[1:, 1:] /= np.sum(new_state_transition[1:, 1:], 1)[:, None]  # normalize
        new_state_transition[0, 0] = 0

        # start transition
        new_state_transition[0, 1:] = np.exp(bw_output.state_p[0, :])
        new_state_transition[0, 1:] /= np.sum(new_state_transition[0, 1:])

        # end transition
        new_state_transition[1:, 0] = np.exp(bw_output.forward[-1, :] + eb_symbols[-1, :]-bw_output.model_p)
        new_state_transition[1:, 0] /= np.sum(new_state_transition[1:, 0])

        new_emission = np.zeros((n_states, n_alphabet))
        posterior = np.exp(bw_output.state_p)  # todo: better idea?
        for sym in range(0, n_alphabet):
            where_sym = (symbol_seq == sym)
            new_emission[1:, sym] = np.sum(posterior[where_sym], 0)

        new_emission /= np.sum(new_emission, 1)[:, None]  # normalize
        new_model = HMMModel.HMMModel(new_state_transition, new_emission)
        prob = bw_output.model_p

    return new_model