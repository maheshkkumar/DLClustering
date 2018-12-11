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

  1. **MNIST** - [Download](https://www.google.com)
  2. **Fashion-MNIST** - [Download](https://www.google.com)
  
 `Note: The above mentioned datasets are included in Keras framework. The hyperlinks can be visited to infer additional 
 information about the datasets`

### Pre-trained models
The pre-trained **autoencoder** models can be downloaded from the hyperlinks displayed below and move the downloaded files to **models** folder.
  1. **MNIST** - 
        2. Without Attention: [Weights](https://myumanitoba-my.sharepoint.com/:u:/g/personal/kumarkm_myumanitoba_ca/EWOwz7Vz6LNMn3G1qmAZsrIB2wvCJUEPRjB6BGnYNtYKLg?e=oiuVfu)
        3. With Attention: [Weights](https://myumanitoba-my.sharepoint.com/:u:/g/personal/kumarkm_myumanitoba_ca/Ebxjxr8fkpFAgjFGzjbQa_UBvhMNKFQg-yvPypXqCkcURw?e=sPKcBc)
  2. **Fashion-MNIST** - 
        2. Without Attention: [Weights](https://myumanitoba-my.sharepoint.com/:u:/g/personal/kumarkm_myumanitoba_ca/EZA2TS8CqTNFkFF8y8Uk1tMBivfSO6vy6qymWiQK-4JUuA?e=fnTvJu)
        3. With Attention: [Weights](https://myumanitoba-my.sharepoint.com/:u:/g/personal/kumarkm_myumanitoba_ca/EY8x_wrvKQJCrEGiShHC5g4BucaJLIa2ufpt5IPRO5ISTQ?e=xW5S93)

Or download the entire models folder: [Download](https://myumanitoba-my.sharepoint.com/:f:/g/personal/kumarkm_myumanitoba_ca/EkTP90yiF0hBr0kl3qXlgAcBV7HWs5IblKJ5Y8s5m6nzbg?e=4G4UYk)

### Parameters to run the script
1. Run ```train.py -h``` to know the additional command line parameters.

        usage: train.py [-h] [-il INCLUDE_LAYER] [-d {mnist,fmnist}] [-bs BATCH_SIZE]
                [-att ATTENTION] [-m {ae,cae}] [-citer CLUSTER_ITERATIONS]
                [-aiter AE_ITERATIONS] [-iu INTERVAL_UPDATION]
                [-tt TOLERANCE_THRESHOLD] [-aew AE_WEIGHTS]
                [-od OUTPUT_DIRECTORY] [-lr LEARNING_RATE]

        optional arguments:
          -h, --help            show this help message and exit
          -il INCLUDE_LAYER, --include_layer INCLUDE_LAYER
                                Include an additional layer in auto encoder
          -d {mnist,fmnist}, --dataset {mnist,fmnist}
                                Name of the dataset
          -bs BATCH_SIZE, --batch_size BATCH_SIZE
                                Size of each batch
          -att ATTENTION, --attention ATTENTION
                                Attention for training
          -m {ae,cae}, --mode {ae,cae}
                                Type of auto encoder model
          -citer CLUSTER_ITERATIONS, --cluster_iterations CLUSTER_ITERATIONS
                                Number of training iterations for the cluster network
          -aiter AE_ITERATIONS, --ae_iterations AE_ITERATIONS
                                Number of training iterations for autoencoder
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

1. **MNIST**: 
    1. Without Attention: 
            ```
            python train.py -d mnist -m cae -aiter 500 -citer 20000 -aew ./models/mnist_ae_without_attention.h5 -lr 1
            ```
    2. With Attention: 
            ```
            python train.py -d mnist -m cae -aiter 500 -citer 20000 -aew ./models/mnist_ae_with_attention.h5 -lr 0.1
            ```
1. **Fashion-MNIST**: 
    1. Without Attention: 
            ```
            python train.py -d fmnist -m cae -aiter 500 -citer 20000 -aew ./models/mnist_ae_without_attention.h5 -lr 1
            ```
    2. With Attention: 
            ```
            python train.py -d fmnist -m cae -aiter 500 -citer 20000 -aew ./models/mnist_ae_with_attention.h5 -lr 0.1
            ```

The script stops on either completing the specified clustering iterations or upon reaching the tolerance level.

<!--
### Visualising the results of clustering

Optimal parameters for the visualization script.

        usage: visualize_results.py [-h] [-r {tsne,barchart,scatterplot}]
                            [-m {ae,dae}] [-d {mnist,fmnist}] -aew AE_WEIGHTS
                            [-att ATTENTION] [-iu INTERVAL_UPDATION]
                            [-od OUTPUT_DIRECTORY]

        optional arguments:
          -h, --help            show this help message and exit
          -r {tsne,barchart,scatterplot}, --result {tsne,barchart,scatterplot}
                                Type of the result visualization and generation
          -m {ae,dae}, --model {ae,dae}
                                Type of the model to be loaded to generate the results
          -d {mnist,fmnist}, --dataset {mnist,fmnist}
                                Choice of the dataset
          -aew AE_WEIGHTS, --ae_weights AE_WEIGHTS
                                Path of the pre-trained auto-encoder weights
          -att ATTENTION, --attention ATTENTION
                                Attention required for training
          -iu INTERVAL_UPDATION, --interval_updation INTERVAL_UPDATION
                                Interval to update the cluster centroid
          -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                                Path of the output directory to store the results and
                                training models

To visualize the clusters, run the following commands:

1. MNIST: ```python visualize_results.py -d mnist -m dae -aew ./models/mnist_ae_model.h5```
1. Fashion-MNIST: `python visualize_results.py -d fmnist -m dae -aew ./models/mnist_ae_model.h5`
-->