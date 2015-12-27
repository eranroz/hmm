import argparse
import os

from hmm_kit import HMMModel
import pandas as pd
import numpy as np
from hmm_kit import bwiter
from sklearn.cluster import KMeans
import pickle as pkl


def train_model(input_path, kernel, states, emission=None, transition=None, model_name=None, html=False):
    input_dir = os.path.dirname(os.path.basename(input_path))
    stop_condition = bwiter.DiffCondition(1)
    if model_name is None:
        model_name = 'Model_{}'.format(input_dir)

    if input_path.endswith('csv'):
        read_input = pd.read_csv(input_path).values
    else:
        read_input = open(input_path, 'r').read()

    if transition:
        initial_transition = pd.read_csv(transition)
        if np.sum(transition, 1) != 1:
            raise Exception('Invalid transition matrix')
    else:
        initial_transition = np.random.random((states, states))
        np.fill_diagonal(initial_transition, initial_transition.sum(1))
        initial_transition /= np.sum(initial_transition, 1)[:, np.newaxis]

    if kernel == 'discrete':
        alphabet = list(sorted(set(read_input)))
        if emission:
            initial_emission = pd.read_csv(emission)
        else:
            initial_emission = np.random.random((states, len(alphabet)))

        model = HMMModel.DiscreteHMM(initial_transition, initial_emission)
    elif kernel == 'gaussian':
        kmean_model = KMeans(n_clusters=states - 1)
        kmean_model.fit(read_input)
        read_clusters = kmean_model.predict(read_input)
        cluster_mean = kmean_model.cluster_centers_
        cluster_cov = [np.cov(read_input[read_clusters == state, :].T) for state in range(states - 1)]
        model = HMMModel.GaussianHMM(initial_transition, [(mean, cov) for mean, cov in zip(cluster_mean, cluster_cov)])
    else:
        raise Exception('Unknown kernel')
    new_model, prob = bwiter.bw_iter(read_input.T, model, stop_condition)
    with open('{}.pkl'.format(model_name), 'wb') as model_pkl:
        pkl.dump(new_model, model_pkl)

    # output html description to file
    if html:
        html_description = new_model.html_state_transition()
        with open('{}.html'.format(model_name), 'w') as model_description_fd:
            model_description_fd.write('<html><body>{}</body></html>'.format(html_description))


def decode_data(model_path, input_path, output_path=None):
    if input_path.endswith('csv'):
        read_input = pd.read_csv(input_path).T.values
    else:
        read_input = open(input_path, 'r').read()

    if not os.path.exists(model_path):
        model_path += '.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    with open(model_path, 'rb') as model_fd:
        model = pkl.load(model_fd)
    decode = model.viterbi(read_input)
    if output_path:
        _, ext = os.path.splitext(output_path)
        if ext == '' or ext == '.npy':
            np.save(output_path, decode)
        elif ext == '.csv':
            pd.DataFrame(decode).to_csv(output_path, index=False)
        else:
            raise Exception('unexpected extension')
    else:
        print(output_path)


def posterior_data(model_path, input_path, output_path=None):
    if input_path.endswith('csv'):
        read_input = pd.read_csv(input_path).T.values
    else:
        read_input = open(input_path, 'r').read()

    if not os.path.exists(model_path):
        model_path += '.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    with open(model_path, 'rb') as model_fd:
        model = pkl.load(model_fd)

    fbresult = model.forward_backward(read_input)
    if output_path:
        _, ext = os.path.splitext(output_path)
        if ext == '' or ext == '.npy':
            np.save(output_path, fbresult.state_p)
        elif ext == '.csv':
            pd.DataFrame(fbresult.state_p.T).to_csv(output_path, index=False)
        else:
            raise Exception('unexpected extension')
    else:
        print(output_path)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="", dest='command')
    subparsers.required = True

    create_model_parser = subparsers.add_parser('train_model', help='Creates a HMM model')
    create_model_parser.add_argument('input', help='Path to input file', type=str)
    create_model_parser.add_argument('--kernel',
                                     help='Emission kernel for the model. binary is a spacial type of discrete',
                                     choices=['discrete', 'gaussian'],
                                     default='gaussian')
    create_model_parser.add_argument('--states', help='Number of states. 0 for auto selection', type=int, default=3)

    create_model_parser.add_argument('--emission', type=str, help='Path to file with emission probabilities')
    create_model_parser.add_argument('--transition', help='Path to file of transition probabilities')

    create_model_parser.add_argument('--model_name', help='Name for the model', default=None)
    create_model_parser.set_defaults(
        func=lambda args: train_model(args.input, args.kernel, args.states, args.emission, args.transition,
                                      args.model_name))

    # decode
    decoder_parser = subparsers.add_parser('decode', help='Decode a sequence (find the most probable states)')
    decoder_parser.add_argument('input', help='Path to input file', type=str)
    decoder_parser.add_argument('model', help='path to a model npz file created using train_model')
    decoder_parser.add_argument('output', help='path to output for the decode', default=None)
    decoder_parser.set_defaults(func=lambda args: decode_data(args.model, args.input, args.output))

    # posterior
    posterior_parser = subparsers.add_parser('posterior', help='Finds the posterior of states')
    posterior_parser.add_argument('input', help='Path to input file', type=str)
    posterior_parser.add_argument('model', help='path to a model npz file created using train_model')
    posterior_parser.add_argument('output', help='path to output for the decode', default=None)
    posterior_parser.set_defaults(func=lambda args: posterior_data(args.model, args.input, args.output))

    command_args = parser.parse_args()
    command_args.func(command_args)


if __name__ == '__main__':
    main()
