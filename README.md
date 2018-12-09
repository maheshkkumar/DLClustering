## Deep Latent Feature Clustering

A method to employ a neural network to learn the latent representation of the multimedia data (e.g., images) to better form the clusters in the lower dimensional clustering friendly space.

This code implements the unsupervised learning mechanism using Convolutional Neural Network (ConvNet) as described in the paper. Additionally, this repository includes the following procedures: 
    
  1. Training the autoencoder.
  2. Loading a pre-trained autoencoder to train the `k-means` cluster in the ConvNet.
  3. Visualize the results using T-SNE method.
    

### Steps to setup the project (requirements)

1. Clone or download the repository.
2. Setup the `python 2.7` environment (either using virtualenv or anaconda distribution)
    1. `Virtualenv`
        1. Create a new environment: `virtualenv -p ~/path/python2.7 project_name`
    2. `Conda`
        1. Create a new environment: `conda create -n python=2.7 project_name`
3. Activate the environment:
    1. Windows: `source project_name/bin/activate`
    2. Linux / Ubuntu: 
        1. `source ~/.bashrc`
        1. `source activate project_name`
4. Install all the project dependencies `pip install -r requirements.txt`

### Datasets

The following datasets were used for evaluating the proposed approach for deep latent feature unsupervised learning.

  1. MNIST - [Download](https://www.google.com)
  2. Fashion-MNIST - [Download](https://www.google.com)
  
 `Note: The above mentioned datasets are included in Keras framework. The hyperlinks can be visited to infer additional 
 information about the datasets`

### Pre-trained models
The pre-trained autoencoder models can be downloaded from the hyperlinks displayed below:
  1. MNIST - [Download](https://www.google.com)
  2. Fashion-MNIST - [Download](https://www.google.com)
  
Move the downloaded pre-trained autoencoder models to `models` folder.

### Parameters to run the script
1. Run `train.py -h` to know the additional command line parameters.

        usage: train.py [-h] [-il INCLUDE_LAYER]
                [-d {mnist,fmnist,stl,cifar10,usps,coil20}] [-bs BATCH_SIZE]
                [-att ATTENTION] [-m {ae,dae}] [-citer CLUSTER_ITERATIONS]
                [-aiter AE_ITERATIONS] [-iu INTERVAL_UPDATION]
                [-tt TOLERANCE_THRESHOLD] [-aew AE_WEIGHTS]
                [-od OUTPUT_DIRECTORY] [-lr LEARNING_RATE]

        optional arguments:
          -h, --help            show this help message and exit
          -il INCLUDE_LAYER, --include_layer INCLUDE_LAYER
                                Include an additional layer in auto encoder
          -d {mnist,fmnist,stl,cifar10,usps,coil20}, --dataset {mnist,fmnist,stl,cifar10,usps,coil20}
                                Name of the dataset
          -bs BATCH_SIZE, --batch_size BATCH_SIZE
                                Size of each batch
          -att ATTENTION, --attention ATTENTION
                                Attention for training
          -m {ae,dae}, --mode {ae,dae}
                                Type of auto encoder model
          -citer CLUSTER_ITERATIONS, --cluster_iterations CLUSTER_ITERATIONS
                                Number of training iterations for the cluster network
          -aiter AE_ITERATIONS, --ae_iterations AE_ITERATIONS
                                Number of training iterations for auto encoder
          -iu INTERVAL_UPDATION, --interval_updation INTERVAL_UPDATION
                                Saving model once the interval limit is reached
          -tt TOLERANCE_THRESHOLD, --tolerance_threshold TOLERANCE_THRESHOLD
                                Tolerance threshold to train the cluster network
          -aew AE_WEIGHTS, --ae_weights AE_WEIGHTS
                                Weights of pre-trained autoencoder
          -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                                Path of the output directory to store the results and
                                training models
          -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                                Learning rate for the experiment
                                
### Using a pre-trained autoencoder to train the clustering

Use the downloaded pre-trained autoencoder to start the clustering task.

1. MNIST: `python train.py -d mnist -m dae -aiter 500 -citer 20000 -aew ./models/mnist_ae_model.h5 -lr 1`
1. Fashion-MNIST: `python train.py -d fmnist -m dae -aiter 500 -citer 20000 -aew ./models/mnist_ae_model.h5 -lr 1`

The script stops on either completing the specified clustering iterations or upon reaching the tolerance level.

### Visualising the results of clustering

To visualize the clusters, run the following commands:

1. MNIST: `python visualize.py -d mnist -m dae -aew ./models/mnist_ae_model.h5`
1. Fashion-MNIST: `python visualize.py -d fmnist -m dae -aew ./models/mnist_ae_model.h5`