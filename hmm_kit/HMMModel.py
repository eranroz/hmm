__author__ = 'eranroz'
import scipy.stats
import numpy as np
from abc import ABCMeta, abstractmethod


class HMMModel(object):
    """
    base model for HMM
    """
    __metaclass__ = ABCMeta

    def __init__(self, state_transition, emission):
        """
        Initializes a new HMM model.
        @param state_transition: state transition matrix.
                    (A & Pi in Rabiner's paper)
                    with rows - source state, cols - target state.
                    0 state assumed to be the begin state
                    (according to Durbin's book)
        @param emission: observation symbol probability distribution
                        (B in Rabiner's paper)
                        rows - states, cols - output
                        The begin state should have emission too (doesn't matter what)
        """
        self.state_transition = state_transition
        self.emission = emission

    def num_states(self):
        """
        Get number of states in the model
        """
        return self.state_transition.shape[0]

    def num_alphabet(self):
        """
        Get number of symbols in the alphabet
        """
        return self.emission.shape[1]

    def get_state_transition(self):
        """
        State transition matrix: rows - source state, cols - target state
        """
        return self.state_transition

    def get_emission(self):
        """
        Emission matrix:  rows states, cols - symbols
        """
        return self.emission

    @abstractmethod
    def _maximize_emission(self, seq, gammas):
        """
        part of the maximization step of the EM algorithm (Baum-Welsh)
        should update the emission probabilities according to the forward-backward results

        @param seq symbol sequence
        @param gammas from backward forward
        """
        pass

    def _maximize_transition(self, seq, bf_output):
        """
        part of the maximization step of the EM algorithm (Baum-Welsh)
        should state transition probabilities according to the forward-backward results

        @param seq: observation sequence
        @param bf_output: output of the forward-backward algorithm
        @return:
        """
        new_state_transition = self.state_transition.copy()
        back_emission_seq = np.array([self.get_emission()[1:, s] for s in seq])
        back_emission_seq *= bf_output.backward / bf_output.scales[:, None]

        new_state_transition[1:, 1:] *= np.dot(bf_output.forward[:-1, :].transpose(), back_emission_seq[1:, :])
        new_state_transition[1:, 1:] /= np.sum(new_state_transition[1:, 1:], 1)[:, None]  # normalize

        # start transition
        new_state_transition[0, 1:] = bf_output.state_p[0, :]
        new_state_transition[0, 1:] /= np.sum(new_state_transition[0, 1:])

        # end transition
        new_state_transition[1:, 0] = bf_output.forward[-1, :] * back_emission_seq[-1, :] / bf_output.scales[-1]
        new_state_transition[1:, 0] /= np.sum(new_state_transition[1:, 0])
        #update transition matrix
        self.state_transition = new_state_transition

    def maximize(self, seq, bw_output):
        """
        Maximization step for in Baum-Welsh algorithm (EM)

        @param seq symbol sequence
        @param bw_output results of backward forward (scaling version)
        """
        self._maximize_transition(seq, bw_output)
        self._maximize_emission(seq, bw_output.state_p)

    def __str__(self):
        return '\n'.join(
            ['Model parameters:', 'Emission:', str(self.emission), 'State transition:',
             str(self.state_transition)])


class DiscreteHMM(HMMModel):
    """
    Discrete Hidden Markov Model

     Handles sequences with discrete alphabet
    """

    def _maximize_emission(self, seq, gammas):
        new_emission_matrix = np.zeros((self.num_states(), self.num_alphabet()))

        state_p = gammas
        for sym in range(0, self.num_alphabet()):
            where_sym = (seq == sym)
            new_emission_matrix[1:, sym] = np.sum(state_p[where_sym, :], 0)

        # normalize
        new_emission_matrix[1:, :] /= np.sum(new_emission_matrix[1:, :], 1)[:, None]

        self.emission = new_emission_matrix


class ContinuousHMM(HMMModel):
    """
    Continuous HMM for observations of real values

    The states are gaussian (or gaussian mixtures)
    @param state_transition: state transition matrix
    @param mean_vars: array of mean, var tuple (or array of such for mixtures)
    @param emission_density: log-concave or elliptically symmetric density
    @param mixture_coef: mixture coefficients
    """

    def __init__(self, state_transition, mean_vars, emission_density=scipy.stats.norm, mixture_coef=None):
        emission = _ContinuousEmission(mean_vars, emission_density, mixture_coef)
        super().__init__(state_transition, emission)

    def _maximize_emission(self, seq, gammas):
        mean_vars = np.zeros((self.num_states(), 2))

        if self.emission.mixtures is None:
            state_norm = np.sum(gammas, 0)
            mu = np.sum(gammas * seq[:, None], 0) / state_norm
            sym_min_mu = np.power(seq[:, None] - mu, 2)
            std = np.sqrt(np.sum(gammas * sym_min_mu, 0) / state_norm)
            min_std = 1e-10
            std = np.maximum(std, min_std)  # it must not get to zero
            mean_vars[1:, :] = np.column_stack([mu, std])
            self.emission = _ContinuousEmission(mean_vars, self.emission.dist_func)
        else:  # TODO: not yet fully tested
            mean_vars = [(0, 0)]
            mixture_coeff = [1]
            for state in np.arange(0, self.num_states() - 1):
                has_coeff = True
                try:
                    if len(self.emission.mixtures[state + 1]) > 1:
                        coeff_pdfs = [self.emission.dist_func(mean, var).pdf for mean, var in
                                      self.emission.mean_vars[state + 1]]
                        coeff_obs = np.array([[p(s) for p in coeff_pdfs] for s in seq])
                        coeff_obs /= np.sum(coeff_obs, 1)[:, None]
                        gamma_coeff = coeff_obs * gammas[:, state][:, None]
                        seq_man = seq[:, None]
                    else:
                        gamma_coeff = gammas[:, state]
                except TypeError:
                    gamma_coeff = gammas[:, state]
                    seq_man = seq
                    has_coeff = False

                sum_gamma = np.sum(gamma_coeff, 0)

                mu = np.sum(gamma_coeff * seq_man, 0) / sum_gamma
                mu *= self.emission.mixtures[state + 1]
                sym_min_mu = np.power(seq_man - mu, 2)
                std = np.sqrt(np.sum(gamma_coeff * sym_min_mu, 0) / sum_gamma)
                min_std = 1e-10
                std = np.maximum(std, min_std)  # it must not get to zero
                if has_coeff:
                    mean_vars.append(list(zip(mu, std)))
                else:
                    mean_vars.append((mu, std))
                mixture_coeff.append(sum_gamma / np.sum(sum_gamma))

            self.emission = _ContinuousEmission(mean_vars, self.emission.dist_func, mixture_coeff)


class _ContinuousEmission():
    """
    Emission for continuous HMM.
    """

    def __init__(self, mean_vars, dist=scipy.stats.norm, mixture_coef=None):
        """
        Initializes a new continuous distribution states.
        @param mean_vars: np array of mean and variance for each state
        @param dist: distribution function
        @return: a new instance of ContinuousDistStates
        """
        self.dist_func = dist
        self.mean_vars = mean_vars
        self.mixtures = mixture_coef
        self.cache = dict()
        self.min_p = 1e-5
        self.states = self._set_states()
        self._set_states()

    def _set_states(self):
        from functools import partial

        if self.mixtures is None:
            states = ([self.dist_func(mean, var).pdf for mean, var in self.mean_vars])
        else:
            states = []
            for mean_var, mixture in zip(self.mean_vars, self.mixtures):
                try:
                    mix_pdf = [self.dist_func(mean, var).pdf for mean, var in mean_var]
                    #mix = lambda x: _ContinuousEmission.mixture_pdf(mix_pdf, mixture, x)
                    mix = partial(_ContinuousEmission.mixture_pdf, mix_pdf, mixture)

                    if sum(mixture) != 1:
                        raise Exception("Bad mixture - mixture for be summed to 1")
                except TypeError:
                    mix = self.dist_func(mean_var[0], mean_var[1]).pdf
                states.append(mix)
        return states

    @staticmethod
    def mixture_pdf(pdfs, mixtures, val):
        """
        Mixture distrbution
        @param pdfs:
        @param mixtures:
        @param val:
        @return:
        """
        return np.dot([p(val) for p in pdfs], mixtures)

    def __getitem__(self, x):
        if isinstance(x[0], slice):
            try:
                return self.cache[x[1]]
            except KeyError:
                pdfs = np.array([dist(x[1]) for dist in self.states[x[0]]])
                pdfs = np.maximum(pdfs, self.min_p)
                self.cache[x[1]] = pdfs
                return self.cache[x[1]]
        else:
            return self.states[x[0]].pdf(x[1])

    def __getstate__(self):
        return {
            'mean_vars': self.mean_vars,
            'mixture_coef': self.mixtures
        }

    def __setstate__(self, state):
        self.mean_vars = state['mean_vars']
        self.mixtures = state['mixture_coef']
        self.dist_func = scipy.stats.norm  # TODO: save name for the method to support other dist funcs
        self.min_p = 1e-5
        self.cache = dict()
        self.states = self._set_states()

    def __str__(self):
        return '\n'.join([str(self.dist_func.name) + ' distribution', 'Mean\t Var', str(self.mean_vars[1:, :])])