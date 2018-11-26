import os

import numpy as np
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.models import Model
from keras.preprocessing.image import img_to_array, array_to_img
from sklearn.preprocessing import MinMaxScaler

def preprocess_image(img):
    img = array_to_img(img, scale=False).resize((224, 224))
    img = img_to_array(img)
    return img

def vgg16_features(data):
        model = VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
        features_model = Model(model.input, model.get_layer('fc1').output)
        data = np.asarray([preprocess_image(image) for image in data])
        data = preprocess_input(data)
        features = features_model.predict(data)

        return features

def load_data(dataset, mode='train'):
    if dataset == 'mnist':
        (train_input, train_labels), (test_input, test_labels) = mnist.load_data()
        data, labels = (train_input, train_labels) if mode is 'train' else (test_input, test_labels)

        data = data.reshape((data.shape[0], -1))
        data = np.divide(data, 255.)
        print("Loading MNIST dataset: {}".format(data.shape))
        return data, labels

    elif dataset == 'fmnist':
        (train_input, train_labels), (test_input, test_labels) = fashion_mnist.load_data()
        data, labels = (train_input, train_labels) if mode is 'train' else (test_input, test_labels)

        data = data.reshape((data.shape[0], -1))
        data = np.divide(data, 255.)

        print("Loading Fashion MNIST dataset: {}".format(data.shape))
        return data, labels

    elif dataset == 'cifar10':

        data_path = './data/cifar10/{}_features.npy'.format(mode)

        (train_input, train_labels), (test_input, test_labels) = cifar10.load_data()
        data, labels = (train_input, train_labels) if mode is 'train' else (test_input, test_labels)

        labels = labels.reshape((50000,)) if mode is 'train' else labels.reshape(10000)

        if os.path.exists(data_path):
            return np.load(data_path), labels

        features = np.zeros((labels.shape[0], 4096))
        for r in range(6):
            idx = range(r * 10000, (r + 1) * 10000)
            features[idx] = vgg16_features(data[idx])

        features = MinMaxScaler().fit_transform(features)
        np.save('./data/cifar10/{}_features.npy'.format(mode))
        return features, labels

    else:
        return None
