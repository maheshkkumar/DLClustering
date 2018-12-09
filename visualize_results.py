import argparse

import numpy as np
from keras.initializers import VarianceScaling, glorot_uniform
from keras.optimizers import SGD

from helpers.dataset import load_data
from model import ClusteringNetwork

# dataset specific training parameters
dataset_parameters = {
    'mnist': {
        'interval_updation': 150,
        'training_steps': 300,
        'data_initialization': VarianceScaling(scale=(1. / 3.), mode='fan_in', distribution='uniform', seed=1),
        'optimizer': SGD(lr=1, momentum=0.9)
    },
    'fmnist': {
        'interval_updation': 150,
        'training_steps': 300,
        'data_initialization': VarianceScaling(scale=(1. / 3.), mode='fan_in', distribution='uniform', seed=1),
        'optimizer': SGD(lr=1, momentum=0.9)
    },
    'stl': {
        'interval_updation': 30,
        'training_steps': 10,
        'data_initialization': glorot_uniform(seed=1),
        'optimizer': 'adam'
    },
}


class GenerateResults(object):
    """
    Implementation to generate results from the encoder part of autoencoder.
    """

    def __init__(self, args):
        self.result = args.result
        self.ae_mode = args.model
        self.dataset = args.dataset
        self.ae_weights = args.ae_weights
        self.attention = args.attention

        self.train_input, self.train_labels = load_data(self.dataset, mode=self.ae_mode)
        num_clusters = len(np.unique(self.train_labels))
        data_initialization = dataset_parameters[self.dataset]['data_initialization']
        auto_encoder_optimzer = dataset_parameters[self.dataset]['optimizer']
        interval_updation = dataset_parameters[self.dataset][
            'interval_updation'] if args.interval_updation is None else args.interval_updation
        temperature = 1.

        if self.ae_mode == "ae":
            dimensions = [self.train_input.shape[-1], 500, 500, 2000,
                          len(np.unique(self.train_labels))] if args.include_layer is None else [
                self.train_input.shape[-1], 500,
                500,
                2000, args.include_layer,
                len(np.unique(
                    self.train_labels))]
        else:
            dimensions = [32, 64]

        self.model = ClusteringNetwork(dimensions=dimensions, temperature=temperature,
                                       data_initialization=data_initialization,
                                       num_clusters=num_clusters, output_directory=args.output_directory,
                                       dataset=self.dataset,
                                       ae_mode=self.ae_mode, with_attention=self.attention)

        if args.ae_weights:
            self.model.auto_encoder.load_weights(args.ae_weights)
        else:
            self.model.train_auto_encoder(data=self.train_input, labels=self.train_labels,
                                          train_steps=args.ae_iterations,
                                          batch_size=args.batch_size, output_directory=args.output_directory,
                                          optimizer=auto_encoder_optimzer)

    def gr(self):

        result = self.model.encoder.predict(self.train_input)
        print("Type of result variable: {}".format(type(result)))
        print("Shape of result variable: {}".format(result.shape))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--result', help="Type of the result visualization and generation",
                        choices=['tsne', 'barchart', 'scatterplot'])
    parser.add_argument('-m', '--model', help="Type of the model to be loaded to generate the results",
                        choices=['ae', 'dae'])
    parser.add_argument('-d', '--dataset', help="Choice of the dataset", choices=['mnist', 'fmnist'])
    parser.add_argument('-aew', '--ae_weights', help="Path of the pre-trained auto-encoder weights", required=True)
    parser.add_argument('-att', '--attention', help="Attention required for training", default=False)
    parser.add_argument('-iu', '--interval_updation', help="Interval to update the cluster centroid", default=140,
                        type=int)
    parser.add_argument('-od', '--output_directory',
                        help="Path of the output directory to store the results and training models",
                        default="./results/visuals")

    args = parser.parse_args()
    gr = GenerateResults(args)

    gr.gr()
