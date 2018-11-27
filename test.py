import argparse
import time

import numpy as np
from keras.initializers import VarianceScaling
from keras.optimizers import SGD
from tensorflow import set_random_seed
from keras.models import Model
from helpers.compute_accuracy import EvaluatePerformance
from helpers.dataset import load_data
from model import ClusteringNetwork, CustomCluster, AutoEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# seeding values for reproducability
np.random.seed(1)
set_random_seed(1)

# dataset specific training parameters
dataset_parameters = {
    'mnist': {
        'interval_updation': 150,
        'training_steps': 300,
        'data_initialization': VarianceScaling(scale=(1. / 3.), mode='fan_in', distribution='uniform'),
        'optimizer': SGD(lr=1, momentum=0.9)
    },
    'fmnist': {
        'interval_updation': 150,
        'training_steps': 300,
        'data_initialization': VarianceScaling(scale=(1. / 3.), mode='fan_in', distribution='uniform'),
        'optimizer': SGD(lr=1, momentum=0.9)
    },
    'stl': {
        'interval_updation': 30,
        'training_steps': 10,
        'data_initialization': 'glorot_uniform',
        'optimizer': 'adam'
    },
}


def train(args):
    """This method trains the clustering network from scratch if there is no pre-trained auto-encoder, else it will load
    the existing pre-trained auto-encoder to retrieve the latent representation of the images to train the final
    clustering layer in the convolutional neural network.

    Args:
        param1 (args): Command line parameters from argparse

    Returns:
        None

    """

    test_input, test_labels = load_data(args.dataset, mode='test')
    train_input, train_labels = load_data(args.dataset, mode='train')
    dimensions = [train_input.shape[-1], 500, 500, 2000, len(np.unique(train_labels))]
    # auto_encoder, encoder = AutoEncoder().ae(dimensions)
    # custom_cluster = CustomCluster(len(np.unique(data_labels)), name="custom_clusters")(encoder.output)
    # model = Model(inputs=encoder.input, outputs=custom_cluster)
    # model.load_weights(args.cl_weights)

    model = ClusteringNetwork(dimensions=dimensions, temperature=1.0, data_initialization='glorot_normal',
                              num_clusters=len(np.unique(train_labels)), output_directory=args.output_directory)

    model.auto_encoder.load_weights(args.ae_weights)
    # model.load_weights(args.cl_weights)
    # p_labels = model.predict(data_input)

    print("Training KMeans")
    kmeans = KMeans(n_clusters=len(np.unique(train_input)), n_jobs=10)
    train_p_labels = kmeans.fit_predict(train_input)
    print("Trained KMeans!")

    test_p_labels = kmeans.predict(test_input)
    print("NMI: {}".format(normalized_mutual_info_score(test_p_labels, test_labels)))
    # print("P Labels shape: {}".format(p_labels.shape))
    # print("P Labels: {}".format(p_labels))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', help="Name of the dataset", choices=['mnist', 'fmnist', 'stl', 'cifar10'])
    parser.add_argument('-bs', '--batch_size', help="Size of each batch", default=256, type=int)
    parser.add_argument('-citer', '--cluster_iterations', help="Number of training iterations for the cluster network",
                        default=15000, type=int)
    parser.add_argument('-aiter', '--ae_iterations', help="Number of training iterations for auto encoder",
                        default=300, type=int)
    parser.add_argument('-iu', '--interval_updation', help="Saving model once the interval limit is reached",
                        default=140, type=int)
    parser.add_argument('-tt', '--tolerance_threshold', help="Tolerance threshold to train the cluster network",
                        default=0.001, type=float)
    parser.add_argument('-aew', '--ae_weights', help="Weights of pre-trained auto-encoder network", default=None)
    parser.add_argument('-cw', '--cl_weights', help="Weights of pre-trained clustering network", default=None)
    parser.add_argument('-od', '--output_directory',
                        help="Path of the output directory to store the results and training models",
                        default="./results")
    args = parser.parse_args()

    train(args)
