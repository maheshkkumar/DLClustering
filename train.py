#! /usr/bin/env python

"""
The :mod:`train` module implements training pipeline for deep latent feature clustering.
"""

import argparse
import time

import numpy as np
from keras.initializers import VarianceScaling, glorot_uniform
from keras.optimizers import SGD
from sklearn.decomposition import PCA
from tensorflow import set_random_seed

from helpers.compute_accuracy import EvaluatePerformance
from helpers.dataset import load_data
from model import ClusteringNetwork

# seeding values for reproducibility
np.random.seed(1)
set_random_seed(1)

# dataset specific training parameters
dataset_parameters = {
    'mnist': {
        'interval_updation': 150,
        'training_steps': 300,
        'data_initialization': VarianceScaling(scale=(1. / 3.), mode='fan_in', distribution='uniform', seed=1)
    },
    'fmnist': {
        'interval_updation': 150,
        'training_steps': 300,
        'data_initialization': VarianceScaling(scale=(1. / 3.), mode='fan_in', distribution='uniform', seed=1)
    },
    'usps': {
        'interval_updation': 150,
        'training_steps': 300,
        'data_initialization': VarianceScaling(scale=(1. / 3.), mode='fan_in', distribution='uniform', seed=1)
    },
    'cifar10': {
        'interval_updation': 150,
        'training_steps': 500,
        'data_initialization': VarianceScaling(scale=(1. / 3.), mode='fan_in', distribution='uniform', seed=1)
    },
    'coil20': {
        'interval_updation': 150,
        'training_steps': 500,
        'data_initialization': VarianceScaling(scale=(1. / 3.), mode='fan_in', distribution='uniform', seed=1)
    },
    'stl': {
        'interval_updation': 30,
        'training_steps': 10,
        'data_initialization': glorot_uniform(seed=1),
        'optimizer': 'adam'
    },
}


def train(args):
    """This method trains the clustering network from scratch if there is no pre-trained autoencoder, else it will load
    the existing pre-trained autoencoder to retrieve the latent representation of the images to train the final
    clustering layer in the convolutional neural network.

    Args:
        args: Command line parameters from argparse

    Returns:
        None

    """
    dataset = args.dataset
    ae_mode = args.mode
    train_input, train_labels = load_data(dataset, mode=ae_mode)
    num_clusters = len(np.unique(train_labels))
    data_initialization = dataset_parameters[dataset]['data_initialization']
    with_attention = args.attention
    interval_updation = dataset_parameters[dataset][
        'interval_updation'] if args.interval_updation is None else args.interval_updation
    temperature = 1.
    auto_encoder_optimizer = SGD(lr=args.learning_rate, momentum=0.9)

    if ae_mode == "ae":
        if train_input.shape[-1] > 1024:
            print("Shape of training data before transformation: {}".format(train_input.shape))
            train_input = PCA(n_components=728).fit_transform(train_input)
            print("Shape of training data after transformation: {}".format(train_input.shape))

        dimensions = [train_input.shape[-1], 500, 500, 2000,
                      len(np.unique(train_labels))] if args.include_layer is None else [train_input.shape[-1], 500, 500,
                                                                                        2000, args.include_layer,
                                                                                        len(np.unique(train_labels))]
    else:
        dimensions = [32, 64]

    model = ClusteringNetwork(dimensions=dimensions, temperature=temperature, data_initialization=data_initialization,
                              num_clusters=num_clusters, output_directory=args.output_directory, dataset=dataset,
                              ae_mode=ae_mode, with_attention=with_attention)

    if args.ae_weights:
        model.auto_encoder.load_weights(args.ae_weights)
    else:
        model.train_auto_encoder(data=train_input, labels=train_labels, train_steps=args.ae_iterations,
                                 batch_size=args.batch_size, output_directory=args.output_directory,
                                 optimizer=auto_encoder_optimizer)

    model.model.summary()

    start_time = time.time()

    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    p_labels = model.train_cluster_network(data=train_input, labels=train_labels,
                                           tolerance_threshold=args.tolerance_threshold,
                                           iterations=args.cluster_iterations, batch_size=args.batch_size,
                                           interval_updation=interval_updation)

    stop_time = time.time()
    print("Accuracy: {}".format(EvaluatePerformance.accuracy(train_labels, p_labels)))
    print("Time taken to finish the training: {}s".format((stop_time - start_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-il', '--include_layer', help="Include an additional layer in auto encoder", default=None,
                        type=int)
    parser.add_argument('-d', '--dataset', help="Name of the dataset",
                        choices=['mnist', 'fmnist'])
    parser.add_argument('-bs', '--batch_size', help="Size of each batch", default=256, type=int)
    parser.add_argument('-att', '--attention', help="Attention for training", default=False, type=bool)
    parser.add_argument('-m', '--mode', help="Type of auto encoder model", choices=["ae", "cae"])
    parser.add_argument('-citer', '--cluster_iterations', help="Number of training iterations for the cluster network",
                        default=15000, type=int)
    parser.add_argument('-aiter', '--ae_iterations', help="Number of training iterations for autoencoder",
                        default=300, type=int)
    parser.add_argument('-iu', '--interval_updation', help="Saving model once the interval limit is reached",
                        default=140, type=int)
    parser.add_argument('-tt', '--tolerance_threshold', help="Tolerance threshold to train the cluster network",
                        default=0.001, type=float)
    parser.add_argument('-aew', '--ae_weights', help="Weights of pre-trained autoencoder", default=None)
    parser.add_argument('-od', '--output_directory',
                        help="Path of the output directory to store the results and training models",
                        default="./results")
    parser.add_argument('-lr', '--learning_rate', help="Learning rate for the experiment", default=None, type=float)
    args = parser.parse_args()

    train(args)
