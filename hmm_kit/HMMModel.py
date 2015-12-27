from abc import ABCMeta, abstractmethod
from collections import namedtuple
from math import fsum

import numpy as np

from . import _hmmc
from .multivariatenormal import MultivariateNormal, MixtureModel


class HMMModel(object):
    """
    base model for HMM
    """
    __metaclass__ = ABCMeta

    def __init__(self, state_transition, emission, min_alpha=None):
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

        @param min_alpha: prior on the sequence length (transition to itself)
        """

        self.state_transition = state_transition
        self.emission = emission
        self.min_alpha = min_alpha

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

    def collect_stats(self, seq, bf_output):
        """
        Collect statistics based on the expectations, for later use in the maximization step
        @param seq: observed sequence
        @param bf_output: backward forward output
        @return: statistics on transitions, statistics on emission
        """
        transition_stats = self._collect_transition_stats(seq, bf_output)
        emission_stats = self._collect_emission_stats(seq, bf_output.state_p)
        return transition_stats, emission_stats

    def maximize_using_stats(self, transition_stats, emission_stats):
        """
        Use collected statistics to maximize the model.
        @param transition_stats: statistics of the transitions
        @param emission_stats: statistics on emission
        """
        self._maximize_transition_stats(transition_stats)
        self._maximize_emission_stats(emission_stats)

    @abstractmethod
    def _collect_emission_stats(self, seq, gammas):
        """
        collects statistics from backward forward iteration (without normalization) about emission
        @param seq: observation sequence
        @param gammas: matrix of states to probabilities
        @return:
        """
        pass

    def _maximize_transition_stats(self, transition_stats, prob_w):
        # == maximize transition ==

        prob_w = np.ones_like(prob_w)  # replace prob_w with ones - all are equal (it is not normalized)
        transition_stats = np.average(transition_stats, 0, prob_w)
        transition_stats[1:, 1:] /= np.sum(transition_stats[1:, 1:], 1)[:, None]  # normalize
        if self.min_alpha is not None:
            n_states = transition_stats.shape[0] - 1  # minus begin state
            diagonal_selector = np.eye(n_states, dtype='bool')
            self_transitions = transition_stats[1:, 1:][diagonal_selector]
            n_self_transitions = np.maximum(self.min_alpha, self_transitions)
            # reduce the diff from the rest of transitions equally
            transition_stats[1:, 1:][~diagonal_selector] -= (n_self_transitions - self_transitions) / (n_states - 1)
            transition_stats[1:, 1:][diagonal_selector] = n_self_transitions

        # start transition
        transition_stats[0, 1:] /= np.sum(transition_stats[0, 1:])

        # end transition
        transition_stats[1:, 0] /= np.sum(transition_stats[1:, 0])
        # update transition matrix
        self.state_transition = transition_stats

    @abstractmethod
    def _maximize_emission_stats(self, emission_stats, prob_w):
        """
        collects statistics from backward forward iteration (without normalization) about emission
        @param emission_stats: statistics for emission of each state
        @return:
        """
        pass

    def _collect_transition_stats(self, seq, bf_output):
        """
        collects statistics from backward forward iteration (without normalization) about transitions
        @param seq: observation sequence
        @param bf_output: output of backward forward iteration
        @return:
        """

        # collect statistics for the transition matrix
        new_state_transition = self.state_transition.copy()
        emission = self.get_emission()
        back_emission_seq = emission[1:, seq].T
        back_emission_seq *= bf_output.backward / bf_output.scales[:, None]

        new_state_transition[1:, 1:] *= np.dot(bf_output.forward[:-1, :].transpose(), back_emission_seq[1:, :])
        # we avoid normalization in the collection phase:
        # new_state_transition[1:, 1:] /= np.sum(new_state_transition[1:, 1:], 1)[:, None]

        # start transition
        # new_state_transition[0, 1:] = bf_output.state_p[0, :]
        new_state_transition[0, 1:] *= back_emission_seq[0, :]
        new_state_transition[0, 1:] /= np.sum(new_state_transition[0, 1:])

        # end transition
        new_state_transition[1:, 0] = bf_output.forward[-1, :] * back_emission_seq[-1, :] / bf_output.scales[-1]
        # we normalize it although it is collect stats because there must be one
        new_state_transition[1:, 0] /= np.sum(new_state_transition[1:, 0])
        return new_state_transition

    def _maximize_transition(self, seq, bf_output):
        """
        part of the maximization step of the EM algorithm (Baum-Welsh)
        should state transition probabilities according to the forward-backward results

        @param seq: observation sequence
        @param bf_output: output of the forward-backward algorithm
        @return:
        """
        new_state_transition = self.state_transition.copy()
        emission = self.get_emission()
        back_emission_seq = emission[1:, seq].T
        back_emission_seq *= bf_output.backward / bf_output.scales[:, None]

        new_state_transition[1:, 1:] *= np.dot(bf_output.forward[:-1, :].transpose(), back_emission_seq[1:, :])
        new_state_transition[1:, 1:] /= np.sum(new_state_transition[1:, 1:], 1)[:, None]  # normalize
        if self.min_alpha is not None:
            n_states = new_state_transition.shape[0] - 1  # minus begin state
            diagonal_selector = np.eye(n_states, dtype='bool')
            self_transitions = new_state_transition[1:, 1:][diagonal_selector]
            n_self_transitions = np.maximum(self.min_alpha, self_transitions)
            # reduce the diff from the rest of transitions equally
            new_state_transition[1:, 1:][~diagonal_selector] -= (n_self_transitions - self_transitions) / (n_states - 1)
            new_state_transition[1:, 1:][diagonal_selector] = n_self_transitions

        # start transition
        new_state_transition[0, 1:] = bf_output.state_p[0, :]
        # new_state_transition[0, 1:] *= back_emission_seq[0, 1:]
        new_state_transition[0, 1:] /= np.sum(new_state_transition[0, 1:])

        # end transition
        new_state_transition[1:, 0] = bf_output.forward[-1, :] * back_emission_seq[-1, :] / bf_output.scales[-1]
        new_state_transition[1:, 0] /= np.sum(new_state_transition[1:, 0])
        # update transition matrix
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

    def viterbi(self, symbol_seq):
        """
        Find the most probable path through the model

        Dynamic programming algorithm for decoding the states.
        Implementation according to Durbin, Biological sequence analysis [p. 57]

        @param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
        """

        n_states = self.num_states()
        unique_values = set(symbol_seq)
        emission_seq = np.zeros((len(symbol_seq), n_states - 1))
        for v in unique_values:
            emission_seq[symbol_seq == v, :] = np.log(self.get_emission()[1:, v])

        return _hmmc.viterbi(emission_seq, self.state_transition)

    def forward_backward(self, symbol_seq, model_end_state=False, num_stable=False):
        """
        Calculates the probability for the model and each step in it

        @param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
        @param model_end_state: whether to consider end state or not
        @param num_stable: whether to handle numerical stability by changing inf to max number (may cause slowness)
        Remarks:
        this implementation uses scaling variant to overcome floating points errors.
        """
        n_states = self.num_states()
        emission = self.get_emission()

        emission_seq = emission[1:, symbol_seq].T
        state_trans_mat = self.get_state_transition()
        dot = np.dot  # shortcut for performance
        real_transitions_T = state_trans_mat[1:, 1:].T.copy(order='C')
        real_transitions_T2 = state_trans_mat[1:, 1:].copy(order='C')

        # emission * transition
        e_trans = iter(emission_seq[1:, :, np.newaxis] * real_transitions_T)
        forward_iterator = np.nditer([emission_seq, np.newaxis, np.newaxis],
                                     flags=['external_loop', 'reduce_ok'],
                                     op_flags=[
                                         ['readonly'],
                                         ['readwrite', 'allocate', 'no_broadcast'],
                                         ['readwrite', 'allocate', 'no_broadcast']
                                     ],
                                     op_axes=[[-1, 0, 1], [-1, 0, 1], [-1, 0, -1]], order='C')

        # -----	  forward algorithm	  -----
        # initial condition is begin state (in Durbin there is another forward - the begin = 1)
        tup = next(forward_iterator)  # emission_i, forward_i,scaling_i
        try:
            tup[1][...] = np.maximum(state_trans_mat[0, 1:] * tup[0], 1e-100)
        except FloatingPointError:
            tup[1][...] = np.exp(np.maximum(np.log(state_trans_mat[0, 1:]) + np.log(tup[0]), -100))
        tup[2][...] = fsum(tup[1])
        tup[1][...] /= tup[2]
        prev_forward = tup[1]

        # recursion step
        for tup in forward_iterator:
            prev_forward = dot(next(e_trans), prev_forward)

            # scaling - see Rabiner p. 16, or Durbin p. 79
            scaling = tup[2]
            scaling[...] = fsum(prev_forward)  # fsum is more numerical stable
            tup[1][...] = prev_forward = prev_forward / scaling

        forward = forward_iterator.operands[1]
        s_j = forward_iterator.operands[2]
        # end transition
        log_p_model = np.sum(np.log(s_j))
        if model_end_state:  # Durbin - with end state
            end_state = 0  # termination step
            end_transition = forward[emission_seq.shape[0] - 1, :] * state_trans_mat[1:, end_state]
            log_p_model += np.log(sum(end_transition))

        # backward algorithm
        # initial condition is end state
        if model_end_state:
            prev_back = (state_trans_mat[1:, 0])  # Durbin p.60
        else:
            prev_back = np.ones(n_states - 1)  # Rabiner p.7 (24)

        backward_iterator = np.nditer([emission_seq[:0:-1], np.newaxis],  # / s_j[:, None]
                                      flags=['external_loop', 'reduce_ok'],
                                      op_flags=[['readonly'],
                                                ['readwrite', 'allocate']],
                                      op_axes=[[-1, 0, 1], [-1, 0, 1]], order='C')

        # recursion step
        e_trans = iter((emission_seq / s_j[:, np.newaxis])[:0:-1, np.newaxis, :] * real_transitions_T2)
        # keep the scale if we get out of numerical boundaries (may occur if the transition matrix has 0)
        if num_stable:
            for tup in backward_iterator:  # emission_i/scale_i, backward_i
                dot(next(e_trans), prev_back, tup[1])  # = dot(real_transitions_T2, prev_back * tup[0])
                prev_back = tup[1][...] = np.nan_to_num(prev_back)
        else:
            for tup in backward_iterator:  # emission_i/scale_i, backward_i
                dot(next(e_trans), prev_back, tup[1])  # = dot(real_transitions_T2, prev_back * tup[0])

        if model_end_state:
            backward = np.append(backward_iterator.operands[1][::-1], state_trans_mat[1:, 0][np.newaxis, :],
                                 axis=0)  # Durbin p.60
        else:
            backward = np.append(backward_iterator.operands[1][::-1], np.ones((1, n_states - 1)),
                                 axis=0)  # Rabiner p.7 (24)

        # return bf_result
        bf_result = namedtuple('BFResult', 'model_p state_p forward backward scales')
        return bf_result(log_p_model, backward * forward, forward, backward, s_j)

    def forward_backward_log(self, symbol_seq, model_end_state=False):
        """
        Calculates the probability for the model and each step in it

        @param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
        @param model: an HMM model to calculate on the given symbol sequence
        @param model_end_state: whether to consider end state or not

        Remarks:
        this implementation uses log variant to overcome floating points errors instead of scaling.
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

        l_emission = np.log(self.get_emission()[1:, :])
        forward = np.zeros((len(symbol_seq), self.num_states() - 1))
        backward = np.zeros((len(symbol_seq), self.num_states() - 1))

        l_state_transition = np.log(self.get_state_transition()[1:, 1:])
        l_t_state_transition = l_state_transition.transpose()
        # forward algorithm
        # initial condition is begin state (in Durbin there is another forward - the begin = 1)
        prev_forward = forward[0, :] = np.log(self.get_state_transition()[0, 1:]) + l_emission[:, symbol_seq[0]]
        # recursion step
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
            log_p_model = forward[len(symbol_seq) - 1, :] + np.log(self.get_state_transition()[1:, end_state])
            log_p_model = interpolate(np.array([log_p_model]))
        else:
            log_p_model = interpolate([forward[len(symbol_seq) - 1, :]])

        # backward algorithm
        last_index = len(symbol_seq) - 1
        if model_end_state:
            prev_back = backward[last_index, :] = np.log(self.get_state_transition()[1:, 0])
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

    def html_state_transition(self):
        """
        nice html representation (as table) of state transition matrix
        """
        import matplotlib as mpl
        import matplotlib.cm as cm
        backgrounds = cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=np.min(self.state_transition), vmax=np.max(self.state_transition)),
            cmap=cm.Blues)
        color_mapper = lambda x: 'rgb(%i, %i, %i)' % backgrounds.to_rgba(x, bytes=True)[:3]

        cells_ths = ''.join(['<th>%i</th>' % i for i in np.arange(1, self.state_transition.shape[0])])
        states_trs = []
        for state_i, state_trans in enumerate(self.state_transition[1:, 1:]):
            state_description = "<td style=\"font-weight:bold;\">%i</td>" % (state_i + 1)
            state_description += ''.join(['<td style="color:#fff;background:%s">%.2f</td>' % (color_mapper(val), val)
                                          for val in state_trans])

            states_trs.append('<tr>%s</tr>' % state_description)

        template = """
<table style="font-size:85%;text-align:center;border-collapse:collapse;border:1px solid #aaa;" cellpadding="5" border="1">
<tr style="font-size:larger; font-weight: bold;">
    <th>State/Cell</th>
    {cells_ths}
</tr>
{states_trs}
</table>
"""
        return template.format(**({'cells_ths': cells_ths,
                                   'states_trs': '\n'.join(states_trs)}))


class DiscreteHMM(HMMModel):
    """
    Discrete Hidden Markov Model

     Handles sequences with discrete alphabet
    """

    def _collect_emission_stats(self, seq, gammas):
        new_emission_matrix = np.zeros((self.num_states() - 1, self.num_alphabet()))  # except begin state

        state_p = gammas
        for sym in range(0, self.num_alphabet()):
            where_sym = (seq == sym)
            new_emission_matrix[:, sym] = np.sum(state_p[where_sym, :], 0)

        return new_emission_matrix

    def _maximize_emission_stats(self, emission_stats, prob_w):
        # normalize
        prob_w = np.ones(prob_w.shape[0])  # replace to 1 - all equals (non normalized)
        emission_stats = np.average(emission_stats, 0, prob_w)
        emission_stats /= np.sum(emission_stats, 1)[:, None]
        self.emission[1:, :] = emission_stats

    def _maximize_emission(self, seq, gammas):
        new_emission_matrix = self._collect_emission_stats(seq, gammas)

        # normalize
        new_emission_matrix /= np.sum(new_emission_matrix, 1)[:, None]

        self.emission[1:, :] = new_emission_matrix


class GaussianHMM(HMMModel):
    """
    Gaussian HMM for multidimensional gaussian mixtures. Extension for Continuous HMM above

    The states are gaussian mixtures
    @param state_transition: state transition matrix
    @param mean_vars: array of mean, var tuple (or array of such for mixtures)
    @param mixture_coef: mixture coefficients
    """

    def __init__(self, state_transition, mean_vars, mixture_coef=None, min_alpha=None):
        if len(mean_vars) == len(state_transition):
            mean_vars = mean_vars[1:]  # trim the begin emission

        # if not mixture (only tuple) wrap it with another list
        mean_vars = [[mean_cov] if isinstance(mean_cov, tuple) else mean_cov for mean_cov in mean_vars]

        # same type: all mean vars should be lists
        emission = _GaussianEmission(mean_vars, mixture_coef)
        super().__init__(state_transition, emission, min_alpha=min_alpha)

    def _collect_emission_stats(self, seq, gammas):
        state_norm = np.sum(gammas, 0)
        mean_vars = []
        mixture_coeff = []
        for state in np.arange(0, self.num_states() - 1):
            is_mixture = len(self.emission.mixtures[state]) > 1
            if is_mixture:
                emissions = self.emission.components_emission(state, seq)
                sum_emissions = np.sum(emissions, 0)
                emissions /= sum_emissions[:, None]  # normalize
                gamma_state = emissions * gammas[:, state][:, None]
                del emissions
                mixture_coeff.append(sum_emissions / np.sum(sum_emissions))
            else:
                gamma_state = gammas[:, state][:, None]
                mixture_coeff.append([1])

            covars_mixture = []
            new_means = np.dot(seq, gamma_state).T  # avoid normalization by state_norm[state]).T
            for mixture_i, mixture in enumerate(self.emission.mixtures[state]):
                gamma_c = gamma_state[:, mixture_i]
                old_mean = self.emission.mean_vars[state][mixture_i][0]
                seq_min_mean = seq - old_mean.T
                new_cov = np.dot((seq_min_mean * gamma_c), seq_min_mean.T)  # avoid normalization by state_norm[state])
                covars_mixture.append(new_cov)
            mean_vars.append((new_means, covars_mixture))

        return mean_vars, mixture_coeff, state_norm

    def _maximize_emission_stats(self, emissions_stats, prob_w):
        prob_w = np.ones_like(prob_w)  # we use non normalized terms(normalizing by state_norms)
        # extract means and covariances
        mean_vars, mixture_coeff, state_norms = zip(*emissions_stats)
        state_norms = np.sum(state_norms, 0)
        # extract mixture coefficients
        mixture_coeff = np.array(mixture_coeff)
        new_mixcoeff = []
        min_std = 0.5  # np.finfo(float).eps  #1e-5 #
        new_mean_vars = []
        for state in np.arange(0, self.num_states() - 1):
            new_mixcoeff.append(np.average(mixture_coeff[:, state], 0, prob_w))

            mix_mean_covar = [mean_var_i[state] for mean_var_i in mean_vars]
            mean_state_i, covar_state_i = zip(*mix_mean_covar)

            mean_state_i = np.average(mean_state_i, 0, prob_w)[0] / state_norms[state]
            covar_state_i = np.average(covar_state_i, 0, prob_w)[0] / state_norms[state]

            # the diagonal must be large enough
            np.fill_diagonal(covar_state_i, np.maximum(covar_state_i.diagonal().copy(), min_std))
            new_mean_vars.append([(mean_state_i, covar_state_i)])

        self.emission = _GaussianEmission(new_mean_vars, new_mixcoeff)

    def _maximize_emission(self, seq, gammas):
        min_std = 0.5  # np.finfo(float).eps  #1e-5 #
        state_norm = np.sum(gammas, 0)
        mean_vars = []
        mixture_coeff = []
        for state in np.arange(0, self.num_states() - 1):
            is_mixture = len(self.emission.mixtures[state]) > 1
            if is_mixture:
                emissions = self.emission.components_emission(state, seq)
                sum_emissions = np.sum(emissions, 0)
                emissions /= sum_emissions[:, None]  # normalize
                gamma_state = emissions * gammas[:, state][:, None]
                del emissions
                mixture_coeff.append(sum_emissions / np.sum(sum_emissions))
            else:
                gamma_state = gammas[:, state][:, None]
                mixture_coeff.append([1])

            covars_mixture = []
            new_means = (np.dot(seq, gamma_state) / state_norm[state]).T
            for mixture_i, mixture in enumerate(self.emission.mixtures[state]):
                gamma_c = gamma_state[:, mixture_i]
                old_mean = self.emission.mean_vars[state][mixture_i][0]

                seq_min_mean = seq - old_mean.T
                new_cov = np.dot((seq_min_mean * gamma_c), seq_min_mean.T) / state_norm[state]
                # the diagonal must be large enough
                np.fill_diagonal(new_cov, np.maximum(new_cov.diagonal().copy(), min_std))
                covars_mixture.append(new_cov)
            mean_vars.append(list(zip(new_means, covars_mixture)))

        self.emission = _GaussianEmission(mean_vars, mixture_coeff)

    def viterbi(self, symbol_seq):
        """
        Find the most probable path through the model

        Dynamic programming algorithm for decoding the states.
        Implementation according to Durbin, Biological sequence analysis [p. 57]

        @param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
        """
        emission_seq = np.log(self.get_emission()[1:, symbol_seq]).T
        return _hmmc.viterbi(emission_seq, self.state_transition)

    def __str__(self):
        # handling mixtures isn't handled currently
        means = np.array([x[0][0] for x in self.emission.mean_vars])
        covars = np.array([x[0][1].diagonal() for x in self.emission.mean_vars])

        str_rep = 'GMM. Means:\n'
        str_rep += np.array_str(means, precision=2, suppress_small=True, max_line_width=250).replace('\n\n', '\n')
        str_rep += '\nDiagonals for covariance matrices:\n'
        str_rep += np.array_str(covars, precision=2, suppress_small=True, max_line_width=250).replace('\n\n', '\n')
        str_rep += '\nStates transitions% (begin not shown):\n'
        str_rep += np.array_str(100 * (self.state_transition[1:, 1:]), precision=1, suppress_small=True,
                                max_line_width=250)
        return str_rep


class _GaussianEmission:
    """
    Emission for continuous HMM.
    """

    def __init__(self, mean_vars, mixture_coef=None):
        """
        Initializes a new continuous distribution states.
        @param mean_vars: np array of mean and variance for each state
        @return: a new instance of ContinuousDistStates
        """
        # if no mixture defined set to 1
        if mixture_coef is None:
            mixture_coef = np.ones((len(mean_vars), 1))
        self.mean_vars = _GaussianEmission._normalize_mean_vars(mean_vars)
        self.mixtures = mixture_coef
        self.states, self.pseudo_states = self._set_states()

    @staticmethod
    def _normalize_mean_vars(mean_vars):
        # use numpy arrays
        n_mean_vars = []
        for state in mean_vars:
            state_mean_cov = []
            for mean, cov in state:
                mean = np.array(mean)
                if mean.ndim == 0:
                    mean = mean[None, None]
                if mean.ndim == 1:
                    mean = mean[None]
                cov = np.array(cov)
                if cov.ndim == 0:
                    cov = cov[None, None]
                if mean.ndim == 1:
                    cov = cov[None]
                state_mean_cov.append((mean, cov))
            n_mean_vars.append(state_mean_cov)
        return n_mean_vars

    def _set_states(self):
        states = []
        pseudo_states = []
        for mean_var, mixture in zip(self.mean_vars, self.mixtures):
            if mixture == 1:
                emission = MultivariateNormal(mean_var[0][0], mean_var[0][1])
                pseudo_states.append(emission.pdf)
            else:
                emission = MixtureModel([MultivariateNormal(mean, cov) for mean, cov in mean_var], mixture)
                pseudo_states.append(emission.components_pdf)
            states.append(emission.pdf)

        return states, pseudo_states

    def components_emission(self, states, observations, use_log=False):
        min_p = np.finfo(float).eps if use_log else -np.inf
        if isinstance(states, int):
            return np.maximum(self.pseudo_states[states].log_pdf(observations), min_p, use_log=use_log)
        pdfs = np.array([dist(observations, use_log=True) for dist in self.pseudo_states[states]]).T
        return np.maximum(pdfs, min_p)

    def __getitem__(self, x):
        """
        Get emission for state
        @param x:  first index is state (or slice for all states), second is value or array of values
        @return: p according to pdf
        """
        min_p = np.finfo(float).eps
        if isinstance(x[0], int):
            return np.maximum(self.states[x[0]].pdf(x[1]), min_p)

        pdfs = np.array([dist(x[1]) for dist in self.states])

        return np.maximum(pdfs, min_p)

    def __getstate__(self):
        return {
            'mean_vars': self.mean_vars,
            'mixture_coef': self.mixtures
        }

    def __setstate__(self, state):
        self.mean_vars = state['mean_vars']
        self.mixtures = state['mixture_coef']
        self.states, self.pseudo_states = self._set_states()

    def __str__(self):
        return 'Mean\t Var\n %s' % str(self.mean_vars)


class MultinomialHMM(HMMModel):
    """
    A multinomial HMM supporting multiple features
    @param state_transition: transition matrix between states
    @param emission_matrix: n X m X k matrix, n - state, m - feature , k - emission for char k
    @param min_alpha:
    """

    def __init__(self, state_transition, emission_matrix, min_alpha=None):
        emission = MultinomialEmission(emission_matrix)
        self.num_features = emission_matrix.shape[1]
        super().__init__(state_transition, emission, min_alpha=min_alpha)

    def _maximize_emission(self, seq, gammas):
        """
        num_features = seq.shape[1]
        alphabet_size = self.emission.states_prob.shape[2]
        new_emission_matrix = np.zeros((self.num_states(), num_features, alphabet_size))

        state_p = gammas
        for feature in range(seq.shape[1]):
            for sym in range(0, alphabet_size):
                where_sym = (seq[:, feature] == sym)
                new_emission_matrix[1:, feature, sym] = np.sum(state_p[where_sym, :], 0)

        # normalize
        new_emission_matrix[1:, :, :] /= np.sum(new_emission_matrix[1:, :, :], -1)[:, :, np.newaxis]

        self.emission = MultinomialEmission(new_emission_matrix)
        """
        emission_stats = self._collect_emission_stats(seq, gammas)
        self._maximize_emission_stats(emission_stats)

    def _collect_emission_stats(self, seq, gammas):
        num_features = seq.shape[1]
        alphabet_size = self.emission.states_prob.shape[2]
        new_emission_matrix = np.zeros((self.num_states(), num_features, alphabet_size))

        state_p = gammas
        for feature in range(seq.shape[1]):
            for sym in range(0, alphabet_size):
                where_sym = (seq[:, feature] == sym)
                new_emission_matrix[1:, feature, sym] = np.sum(state_p[where_sym, :], 0)

        return new_emission_matrix

    def _maximize_emission_stats(self, emission_stats, prob_w):
        # normalize
        prob_w = np.ones(prob_w.shape[0])  # replace to 1 - all equals (non normalized)
        emission_stats = np.average(emission_stats, 0, prob_w)
        emission_stats[1:, :, :] /= np.sum(emission_stats[1:, :, :], -1)[:, :, np.newaxis]
        self.emission = MultinomialEmission(emission_stats)

    def num_alphabet(self):
        """
        Number of symbols in alphabet based on teh defined emission (in all the features)

        @return:
        """
        return self.emission.states_prob.shape[1] * self.emission.states_prob.shape[2]

    def viterbi(self, symbol_seq):
        """
        Find the most probable path through the model

        Dynamic programming algorithm for decoding the states.
        Implementation according to Durbin, Biological sequence analysis [p. 57]

        @param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
        """
        emission_seq = np.log(self.get_emission()[1:, symbol_seq]).T
        return _hmmc.viterbi(emission_seq, self.state_transition)


class MultinomialEmission:
    """
    MultinomialEmission assumes multinomial distribution e.g (n!/(n1!n2!))*(p1)^n1*(p2)^n2...
    """

    def __init__(self, states_prob):
        """
        initializes a new instance of MultinomialEmission
        @param states_prob: a matrix of nxmxk where n - number of states and m-feature index, k -emission for char k
        """
        self.states_prob = states_prob

    def __getitem__(self, item):
        """
        Get emission for state
        @param item:  first index is state (or slice for all states),
                      second is value or matrix of values [observations x features]
        @return: p according to pdf
        """
        min_p = np.finfo(float).eps
        if isinstance(item[0], int):
            features_prob = self.states_prob[item[0], :, item[1]]
            p = np.prod(features_prob, -1)
        else:

            # features_prob = self.states_prob[:, :, item[1]]
            # p = np.prod(features_prob, -1)
            features_prob = []
            for f in range(self.states_prob.shape[1]):
                features_prob.append(self.states_prob[item[0], f, item[1][:, f]])
            p = np.prod(features_prob, 0)
        p = np.maximum(p, min_p)
        return p
