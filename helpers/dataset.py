import exceptions
from glob import glob

import h5py
import numpy as np
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.models import Model
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from tensorflow import set_random_seed

# seeding values for reproducibility
np.random.seed(1)
set_random_seed(1)

# USPS data path
USPS_PATH = './datasets/usps/usps.h5'
COIL20_PATH = './datasets/coil20/coil20'


def preprocess_image(img):
    """Module to preprocess an image
    """
    img = array_to_img(img, scale=False).resize((224, 224))
    img = img_to_array(img)
    return img


def vgg16_features(data):
    """Module to extract features from VGG16 network
    """
    model = VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
    features_model = Model(model.input, model.get_layer('fc1').output)
    data = np.asarray([preprocess_image(image) for image in data])
    data = preprocess_input(data)
    features = features_model.predict(data)

    return features


def load_data(dataset, mode='ae'):
    """Module to load the dataset for training mlp or cnn autoencoder
    """
    if dataset == 'mnist':
        (train_input, train_labels), (test_input, test_labels) = mnist.load_data()
        data_input = np.concatenate((train_input, test_input))
        data_labels = np.concatenate((train_labels, test_labels))

        if mode == 'cae':
            image_size = data_input.shape[1]
            data = np.reshape(data_input, [-1, image_size, image_size, 1])
            data = np.divide(data, 255.)
        else:
            data = data_input.reshape((data_input.shape[0], -1))
            data = np.divide(data, 255.)
        print("Loading MNIST dataset: {}".format(data.shape))
        return data, data_labels

    elif dataset == 'fmnist':
        (train_input, train_labels), (test_input, test_labels) = fashion_mnist.load_data()
        data_input = np.concatenate((train_input, test_input))
        data_labels = np.concatenate((train_labels, test_labels))

        if mode == 'cae':
            image_size = data_input.shape[1]
            data = np.reshape(data_input, [-1, image_size, image_size, 1])
            data = np.divide(data, 255.)
        else:
            data = data_input.reshape((data_input.shape[0], -1))
            data = np.divide(data, 255.)
        print("Loading Fashion MNIST dataset: {}".format(data.shape))
        return data, data_labels

    elif dataset == 'cifar10':

        (train_input, train_labels), (test_input, test_labels) = cifar10.load_data()
        data_input = np.concatenate((train_input, test_input))
        data_labels = np.concatenate((train_labels, test_labels))

        if mode == 'cae':
            image_size = data_input.shape[1]
            data = np.reshape(data_input, [-1, image_size, image_size, 3])
            data = np.divide(data, 255.)
        else:
            data = data_input.reshape((data_input.shape[0], -1))
            data = np.divide(data, 255.)
        print("Loading CIFAR10 dataset: {}".format(data.shape))
        return data, data_labels.reshape(-1)

    elif dataset == 'usps':
        with h5py.File(USPS_PATH, 'r') as hf:
            train = hf.get('train')
            train_input = train.get('data')[:]
            print("Shape of train input: {}".format(train_input.shape))
            train_labels = train.get('target')[:]
            test = hf.get('test')
            test_input = test.get('data')[:]
            test_labels = test.get('target')[:]
            data_input = np.concatenate((train_input, test_input))
            data_labels = np.concatenate((train_labels, test_labels))

            if mode == 'cae':
                print("Shape of USPS dataset: {}".format(data_input.shape))
                image_size = 16
                data = np.reshape(data_input, [-1, image_size, image_size, 1])
                data = np.divide(data, 255.)
            else:
                data = data_input.reshape((data_input.shape[0], -1))
                data = np.divide(data, 255.)
            print("Loading USPS dataset: {}".format(data.shape))

        return data, data_labels

    elif dataset == 'coil20':
        try:
            images = sorted(glob(COIL20_PATH + '/*.png'))
            data_input, data_labels = [], []
            for img in images:
                img_label = int(img.split('obj')[-1].split('__')[0])
                image = img_to_array(load_img(img))
                data_input.append(image)
                data_labels.append(img_label)
            data_input = np.asarray(data_input).astype(np.float32)
            data_labels = np.asarray(data_labels)
            assert data_input.shape[0] == data_labels.shape[0]

            if mode == 'ae':
                data_input = np.reshape(data_input, (data_input.shape[0], -1))

            print("Loading COIL20 Dataset: {}".format(data_input.shape))

            return data_input, data_labels
        except exceptions as e:
            print("Exception: {}".format(e.message))

    else:
        return None
