"""
Implementation of Baum-Welch, or re-estimation procedure.
See:
* Durbin p. 64
* Rabiner p. 8 and p. 17
"""

from copy import deepcopy
import time
import numpy as np
from . import HMMModel

_time_profiling = False


class IteratorCondition:
    """
    Iteration condition class for Baum-Welch with limited number of iterations
    which also outputs the likelihood for each iteration
    """

    def __init__(self, count):
        """
        @param count: number of iterations
        """
        self.i = count
        self.start = False
        self.prev_p = None
        self.prev_likelihoods = []

    def __call__(self, *args, **kwargs):
        if self.start:
            self.prev_likelihoods.append(args[0])
            print('\t\t\t====(#: %s) Likelihood:' % self.i, args[0], '=====')
        else:
            self.start = True
        self.i -= 1
        # if self.prev_p is not None and args[0]-self.prev_p < 1e-6:
        #    print('Got to maximum!')
        #    return False
        self.prev_p = args[0]
        return self.i >= 0


class DiffCondition:
    """
    Iteration condition class for Baum-Welch based on diff in likelihood
    which also outputs the likelihood for each iteration
    """

    def __init__(self, threshold=100):
        """
        @param threshold: diff likelihood threshold (e.g if likelihood[i]-likelihood[i-1]<threshold we get out)
        """
        self.count = 0
        self.start = False
        self.prev_p = 0
        self.threshold = threshold
        self.positive_iters = 0
        self.prev_likelihoods = []

    def __call__(self, *args, **kwargs):
        if self.start:
            self.prev_likelihoods.append(args[0])
            print('\t\t\t====(#: %s) Likelihood:' % self.count, args[0], '=====')
            self.count += 1
            if self.prev_p == 0:
                self.prev_p = args[0]
                return True
            diff = (args[0] - self.prev_p)
            self.prev_p = args[0]
            if diff < self.threshold:
                self.positive_iters += 1
            else:
                self.positive_iters = 0
            return self.positive_iters < 3
        else:
            self.start = True
            return True


def _transform_stop_condition(stop_condition):
    if isinstance(stop_condition, int):
        stop_condition = IteratorCondition(stop_condition)
    elif stop_condition is None:
        stop_condition = DiffCondition(100)
    return stop_condition


def bw_iter(symbol_seq, initial_model, stop_condition=3, constraint_func=None):
    """
    Estimates model parameters using Baum-Welch algorithm

    @rtype : tuple(HMMModel, int)
    @param symbol_seq: observations
    @param initial_model: Initial guess for model parameters
    @type initial_model: C{ContinuousHMM} or C{DiscreteHMM} or C{GaussianHMM}
    @param stop_condition: Callable like object/function that takes probability as parameter
    @param constraint_func: function to manipulate the model
    @return: HMM model estimation
    """
    new_model = deepcopy(initial_model)
    stop_condition = _transform_stop_condition(stop_condition)

    prob = float('-inf')
    print('Press ctrl+C for early termination of Baum-Welch training')
    try:
        while stop_condition(prob):
            if _time_profiling:
                bef = time.time()
            bw_output = new_model.forward_backward(symbol_seq)
            if _time_profiling:
                print(time.time() - bef)
            prob = bw_output.model_p
            new_model.maximize(symbol_seq, bw_output)
            if constraint_func is not None:
                constraint_func(new_model)
    except KeyboardInterrupt:
        print('Early termination of EM training')

    return new_model, prob


def bw_iter_multisequence(sequences, initial_model=None, stop_condition=3, constraint_func=None):
    """
    Estimates model parameters using Baum-Welch algorithm

    @rtype : tuple(HMMModel, int)
    @param sequences: observations
    @param initial_model: Initial guess for model parameters
    @type initial_model: C{ContinuousHMM} or C{DiscreteHMM} or C{GaussianHMM}
    @param stop_condition: Callable like object/function that takes probability as parameter
    @param constraint_func: function to manipulate the model
    @return: HMM model estimation
    """
    new_model = deepcopy(initial_model)
    stop_condition = _transform_stop_condition(stop_condition)

    prob = float('-inf')
    print('Press ctrl+C for early termination of Baum-Welch training')
    try:
        while stop_condition(prob):
            if _time_profiling:
                bef = time.time()
            transition_stats, emissions_stats = [], []
            prob = []
            for seq in sequences:
                bw_output = new_model.forward_backward(seq)
                if np.isnan(bw_output.model_p):
                    print('Numerical stability issue')
                    continue
                transition_stats_seq, emission_stats_seq = new_model.collect_stats(seq, bw_output)
                transition_stats.append(transition_stats_seq)
                emissions_stats.append(emission_stats_seq)
                prob.append(bw_output.model_p)

            if _time_profiling:
                print(time.time() - bef)

            prob = np.array(prob)
            new_model._maximize_transition_stats(transition_stats, prob)
            new_model._maximize_emission_stats(emissions_stats, prob)
            prob = np.sum(prob)
            if constraint_func:
                constraint_func(new_model)

    except KeyboardInterrupt:
        print('Early termination of EM training')

    return new_model, prob


def bw_iter_log(symbol_seq, initial_model, stop_condition=None, constraint_func=None):
    """
    Estimates model parameters using Baum-Welch algorithm with log backward forward

    @param stop_condition: stop condition for EM iterations. default is 10
    @param symbol_seq: observations
    @param initial_model: Initial guess for model parameters
    @return: HMM model estimation

    @attention it is better to use scaled version as above. implementation is provided as a reference and for comparing
    to scaled version - but it is less tested, and may insert small inaccuracies
    """
    stop_condition = _transform_stop_condition(stop_condition)
    new_model = deepcopy(initial_model)
    prob = 0
    while stop_condition(prob):
        bw_output = new_model.forward_backward_log(symbol_seq)
        new_model.maximize(symbol_seq, bw_output)
        prob = bw_output.model_p
        if constraint_func:
            constraint_func(new_model)

    return new_model, prob
