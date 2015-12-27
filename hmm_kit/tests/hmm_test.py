import unittest
import numpy as np
import hmm_kit.HMMModel


class HmmTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_viterbi(self):
        """
        Occasionally dishonest casino part 1 [Durbin p.55]
        This test the viterbi algorithm - result should be more or less as in Durbin p. 58
        """
        n_tiles = 35000
        fair = True
        dice = []
        real_dice = []
        for i in range(0, n_tiles):
            if fair:
                fair = np.random.randint(1, 101) <= 95
            else:
                fair = np.random.randint(1, 101) >= 90

            real_dice.append(0 if fair else 1)
            if fair:
                dice.append(np.random.randint(1, 7))
            else:
                dice.append(6 if np.random.randint(0, 2) < 1 else np.random.randint(1, 6))

        # state to state
        a_zero = 0.00000001
        state_transition = np.array([
            [a_zero, 1, a_zero],
            [a_zero, 0.95, 0.05],
            [a_zero, 0.1, 0.9]
        ])

        # satet to outputs

        begin_emission = np.ones(6)

        fair_emission = np.ones(6) / 6  # all equal
        unfair_emission = np.ones(6) / 10  # all equal to 1/10 and 6 to 0.5
        unfair_emission[5] = 0.5

        emission = np.array([begin_emission, fair_emission, unfair_emission])
        model = hmm_kit.HMMModel.HMMModel (state_transition, emission)
        translated_dice = np.array(dice)  # translate to emission alphabet
        translated_dice -= 1
        guess_dice = model.viterbi(translated_dice.flatten())

        correct = np.mean(guess_dice == np.array(real_dice))
        self.assertTrue(correct > 0.6)

    def test_gmm(self):
        mean_a, cov_a = [1, 1.2], np.sqrt(np.array([[0.85, 0.3], [0.3, 1.0]]))
        mean_b, cov_b = [1.3, 1.1], np.sqrt(np.array([[1, 0.1], [0.1, 1.0]]))

        n_tiles = 35000
        is_a = True
        real_states = []
        for i in range(0, n_tiles):
            if is_a:
                is_a = np.random.randint(1, 101) <= 95
            else:
                is_a = np.random.randint(1, 101) >= 99

            real_states.append(0 if is_a else 1)

        real_states = np.array(real_states)
        observations = np.zeros((n_tiles, 2))

        observations[real_states == 0, :] = np.random.multivariate_normal(mean_a, cov_a, np.sum(real_states == 0))
        observations[real_states == 1, :] = np.random.multivariate_normal(mean_b, cov_b, np.sum(real_states == 1))

        model = hmm_kit.HMMModel.GaussianHMM(np.array([
            [0, 1, 0],
            [0, 0.95, 0.05],
            [0, 0.01, 0.99]
        ]), [(mean_a, cov_a), (mean_b, cov_b)])

        inferred_state = model.viterbi(observations.T)

        correct = np.mean(inferred_state == np.array(real_states))
        self.assertTrue(correct > 0.7)


if __name__ == '__main__':
    unittest.main()
