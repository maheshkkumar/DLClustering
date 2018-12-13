#! /usr/bin/env python

"""
The :mod:`model` module implements the training of autoencoder and clustering for deep latent feature clustering.
"""

import csv
import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import callbacks
from keras.engine.topology import Layer, InputSpec
from keras.initializers import glorot_uniform
from keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose, Activation, multiply
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from tensorflow import set_random_seed

from helpers.compute_accuracy import ComputeAccuracyCallback
from helpers.compute_accuracy import EvaluatePerformance
from helpers.utils import check_directory

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# seeding values for reproducibility
np.random.seed(1)
set_random_seed(1)


class CAE():
    """Pipeline to implement the autoencoder based on convolutional neural network.

    Sequentially implements an encoder and a decoder to learn the latent features in the input data.

    Encoder: Transforms data from the high dimensional input space to a lower dimensional latent space.
    Decoder: Reconstructs the input data from the latent features
    """

    def __init__(self):
        self.data_initialization = glorot_uniform(seed=1)

    def ae(self, input_shape, latent_dimension, kernel_size, layers, activation='relu', with_attention=False,
           dataset='mnist'):

        print("Layers: {}".format(layers))

        initial_representation = Input(shape=input_shape, name='encoder_input')
        latent_representation = initial_representation

        # encoder sub-network
        for filters in layers:
            latent_representation = Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=2,
                                           activation=activation,
                                           padding='same')(latent_representation)

        # Shape info needed to build Decoder Model
        shape = K.int_shape(latent_representation)

        # Generate the latent vector
        latent_representation = Flatten()(latent_representation)
        latent_representation = Dense(latent_dimension, name='latent_vector')(latent_representation)

        if with_attention:
            print("Adding attention for training")
            # adding attention for extracting relevant features
            attention_probs = Dense(latent_dimension, activation='softmax', name='attention_vec')(latent_representation)
            latent_representation = multiply([latent_representation, attention_probs])

        # Instantiate Encoder Model
        encoder = Model(initial_representation, latent_representation, name='encoder')
        encoder.summary()

        # Build the Decoder Model
        decoder_initial_representation = Input(shape=(latent_dimension,), name='decoder_input')
        decoder_representation = Dense(shape[1] * shape[2] * shape[3])(decoder_initial_representation)
        decoder_representation = Reshape((shape[1], shape[2], shape[3]))(decoder_representation)

        for filters in layers[::-1]:
            decoder_representation = Conv2DTranspose(filters=filters,
                                                     kernel_size=kernel_size,
                                                     strides=2,
                                                     activation=activation,
                                                     padding='same')(decoder_representation)

        if dataset in ["mnist", "fmnist", "usps"]:
            decoder_representation = Conv2DTranspose(filters=1,
                                                     kernel_size=kernel_size,
                                                     padding='same')(decoder_representation)
        else:
            decoder_representation = Conv2DTranspose(filters=3,
                                                     kernel_size=kernel_size,
                                                     padding='same')(decoder_representation)

        cae_output = Activation('sigmoid', name='decoder_output')(decoder_representation)

        # Instantiate Decoder Model
        decoder = Model(decoder_initial_representation, cae_output, name='decoder')
        decoder.summary()

        # Autoencoder = Encoder + Decoder
        autoencoder = Model(initial_representation, decoder(encoder(initial_representation)), name='autoencoder')
        autoencoder.summary()

        return autoencoder, encoder


class AutoEncoder():
    """Pipeline to implement the autoencoder based on multilayer perceptron.

    Sequentially implements an encoder and a decoder to learn the latent features in the input data.

    Encoder: Transforms data from the high dimensional input space to a lower dimensional latent space.
    Decoder: Reconstructs the input data from the latent features
    """

    def __init__(self):
        self.data_initialization = glorot_uniform(seed=1)

    def ae(self, layers, activation='relu', with_attention=False):

        print("Layers: {}".format(layers))

        num_of_layers = len(layers) - 1
        initial_representation = Input(shape=(layers[0],), name='ae_input')
        latent_representation = initial_representation

        # encoder sub-network
        for idx in range(num_of_layers - 1):
            latent_representation = Dense(layers[idx + 1], activation=activation,
                                          kernel_initializer=self.data_initialization,
                                          name="ae_encoder_{}".format(idx))(latent_representation)

        # latent representation of auto encoder
        latent_representation = Dense(layers[-1], kernel_initializer=self.data_initialization,
                                      name="latent_vector")(latent_representation)

        if with_attention:
            print("Adding attention for training")
            # adding attention for extracting relevant features
            attention_probs = Dense(layers[-1], activation='softmax', name='attention_vec')(latent_representation)
            latent_representation = multiply([latent_representation, attention_probs])

        # decoder sub-network
        decoder_representation = latent_representation
        for idx in range(num_of_layers - 1, 0, -1):
            decoder_representation = Dense(layers[idx], activation=activation,
                                           kernel_initializer=self.data_initialization,
                                           name="ae_decoder_{}".format(idx))(decoder_representation)

        # auto encoder output
        decoder_representation = Dense(layers[0], kernel_initializer=self.data_initialization, name="ae_decoder_0")(
            decoder_representation)

        return Model(inputs=initial_representation, outputs=decoder_representation, name='autoencoder'), Model(
            inputs=initial_representation, outputs=latent_representation, name='encoder')


class CustomCluster(Layer):
    """
    Implements a custom layer to cluster the latent features from the encoder.
    """

    def __init__(self, num_clusters, weights=None, temperature=1.0, **kwargs):
        super(CustomCluster, self).__init__(**kwargs)
        self.data_initialization = glorot_uniform(seed=1)
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.custom_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        input_dimension = input_shape[-1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dimension))
        self.custom_clusters = self.add_weight((self.num_clusters, input_dimension),
                                               initializer=self.data_initialization, name='custom_clusters')
        if self.custom_weights:
            self.set_weights(self.custom_weights)
            del self.custom_weights
        self.built = True

    def call(self, inputs, **kwargs):
        soft_label = 1.0 / (1.0 + (
                K.sum(K.square(K.expand_dims(inputs, axis=1) - self.custom_clusters), axis=2) / self.temperature))
        soft_label **= (self.temperature + 1.0) / 2.0
        soft_label = K.transpose(K.transpose(soft_label) / K.sum(soft_label, axis=1))
        return soft_label

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.num_clusters

    def get_config(self):
        configuration = {'num_clusters': self.num_clusters}
        base_configuration = super(CustomCluster, self).get_config()
        return dict(list(base_configuration.items()) + list(configuration.items()))


class ClusteringNetwork(object):
    """
    Pipleline to train the complete architecture (autoencoder + clustering)
    """

    def __init__(self, **kwargs):
        super(ClusteringNetwork, self).__init__()

        self.dataset = kwargs['dataset']
        self.dimensions = kwargs['dimensions']
        self.layers = len(self.dimensions) - 1
        self.num_clusters = kwargs['num_clusters']
        self.temperature = kwargs['temperature']
        self.with_attention = kwargs['with_attention']
        self.ae_mode = kwargs['ae_mode']
        self.data_initialization = kwargs['data_initialization']
        self.output_directory = os.path.join(kwargs['output_directory'], self.dataset)

        check_directory(self.output_directory)
        folder_count = len(os.listdir(self.output_directory))

        self.results_directory = os.path.join(self.output_directory, 'version_{}'.format(folder_count + 1))
        self.input_shapes = {
            'mnist': [28, 28, 1],
            'fmnist': [28, 28, 1],
            'cifar10': [32, 32, 3],
            'usps': [16, 16, 1],
            'coil20': [128, 128, 3]
        }
        self.input_shape = self.input_shapes[self.dataset]

        if self.ae_mode == 'ae':
            self.auto_encoder, self.encoder = AutoEncoder().ae(self.dimensions, with_attention=self.with_attention)
            model_input, model_output = self.encoder.input, CustomCluster(self.num_clusters, name='custom_clusters')(
                self.encoder.output)
        else:
            self.auto_encoder, self.encoder = CAE().ae(layers=self.dimensions, input_shape=self.input_shape,
                                                       kernel_size=3,
                                                       latent_dimension=self.num_clusters,
                                                       with_attention=self.with_attention, dataset=self.dataset)
            model_input, model_output = self.encoder.get_input_at(0), CustomCluster(self.num_clusters,
                                                                                    name="custom_clusters")(
                self.encoder.get_output_at(0))
        self.model = Model(inputs=model_input, outputs=model_output)

    def train_auto_encoder(self, data, labels=None, loss='mse', optimizer='adam', train_steps=200, batch_size=256):

        check_directory(self.results_directory)

        ae_path = os.path.join(self.results_directory, 'auto_encoder')
        check_directory(ae_path)
        ae_model = os.path.join(ae_path, '{}_ae.h5'.format(self.dataset))
        ae_logger = os.path.join(ae_path, '{}_ae.h5'.format(self.dataset))

        print("Training AutoEncoder")
        self.auto_encoder.compile(optimizer=optimizer, loss=loss)

        output_logging = callbacks.CSVLogger(ae_logger, separator=',')
        custom_callback = [output_logging]

        if labels is not None:
            custom_callback.append(
                ComputeAccuracyCallback(data=data, labels=labels, model=self.model, mode=self.ae_mode))

        self.auto_encoder.fit(data, data, batch_size=batch_size, epochs=train_steps, callbacks=custom_callback)
        self.auto_encoder.save_weights(ae_model)
        self.trained_auto_encoder = True

    def load_weights(self, weights):
        self.load_weights(weights)

    def extract_latent_representation(self, data):
        return self.encoder.predict(data)

    def predict(self, data):
        return self.model.predict(data, verbose=0).argmax(1)

    def compile(self, loss='kld', optimizer='sgd'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def evaluate_model(self, labels, p_labels):

        accuracy = np.round(EvaluatePerformance.accuracy(labels, p_labels), 3)
        nmi = np.round(normalized_mutual_info_score(labels, p_labels), 3)
        ari = np.round(adjusted_rand_score(labels, p_labels), 3)

        return accuracy, nmi, ari

    def train_cluster_network(self, data, labels=None, iterations=15000, batch_size=256, tolerance_threshold=1e-3,
                              interval_updation=100):

        check_directory(self.results_directory)
        cluster_path = os.path.join(self.results_directory, 'clustering')
        check_directory(cluster_path)
        log_file = os.path.join(cluster_path, 'result.csv')

        print("Interval Updating: {}".format(interval_updation))
        interval_limit = int(data.shape[0] / batch_size) * 5
        print("Saving interval: {}".format(interval_limit))

        print("Initializing the default centers for each cluster")
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=20)
        p_labels = kmeans.fit_predict(self.encoder.predict(data))
        previous_p_labels = np.copy(p_labels)
        self.model.get_layer(name="custom_clusters").set_weights([kmeans.cluster_centers_])

        model_logger = open(log_file, 'w')
        log_writer = csv.DictWriter(model_logger, fieldnames=['Iteration', 'Accuracy', 'NMI', 'ARI', 'Loss'])
        log_writer.writeheader()

        loss = 0
        index = 0
        indices = np.arange(data.shape[0])

        for iteration in range(iterations):

            model_path = os.path.join(cluster_path, "{}_{}.h5".format(self.dataset, iteration))

            if iteration % interval_updation == 0:
                soft_labels = self.model.predict(data)
                soft_label_dist = EvaluatePerformance.soft_labels_target_dist(soft_labels=soft_labels)

                p_labels = soft_labels.argmax(1)

                if labels is not None:
                    accuracy, nmi, ari = self.evaluate_model(labels, p_labels)
                    loss = np.round(loss, 3)
                    metrics = dict(Iteration=iteration, Accuracy=accuracy, NMI=ari, ARI=ari, Loss=loss)
                    log_writer.writerow(metrics)
                    print(
                        "Iteration: {}, Accuracy: {:.3f}, NMI: {:.3f}, ARI: {:.3f}".format(iteration, accuracy, nmi,
                                                                                           ari))

                tolerance = np.sum(p_labels != previous_p_labels).astype(np.float32) / p_labels.shape[0]
                previous_p_labels = np.copy(p_labels)

                if iteration > 0 and tolerance < tolerance_threshold:
                    print("Current tolerance value: {}, Tolerance threshold: {}".format(tolerance, tolerance_threshold))
                    print("Tolerance threshold reached, hence stopping training.")
                    model_logger.close()

                    accuracy, nmi, ari = self.evaluate_model(labels, p_labels)
                    print("Evaluation (test) results - Accuracy: {}, NMI: {}, ARI: {}".format(accuracy, nmi, ari))
                    break

            # training the model
            idx = indices[index * batch_size: min((index + 1) * batch_size, data.shape[0])]
            loss = self.model.train_on_batch(x=data[idx], y=soft_label_dist[idx])
            index = index + 1 if ((index + 1) * batch_size) <= data.shape[0] else 0

            if iteration % interval_limit == 0:
                print("Saving the current model to: {}".format(model_path))
                self.model.save_weights(model_path)

            iteration += 1

        model_logger.close()
        print("Saving the final model to: {}".format(model_path))
        self.model.save_weights(model_path)

        accuracy, nmi, ari = self.evaluate_model(labels, p_labels)
        print("Evaluation (test) results - Accuracy: {}, NMI: {}, ARI: {}".format(accuracy, nmi, ari))

        return p_labels
