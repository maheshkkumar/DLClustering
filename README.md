## Deep Latent Feature Clustering

> **Note**: The code for the project can be found on [GitHub](https://github.com/maheshkkumar/DLClustering).

A method to employ a neural network to learn the latent representation of the multimedia data (e.g., images) to better form the clusters in the lower dimensional clustering friendly space.

This code implements the unsupervised learning mechanism using Convolutional Neural Network (ConvNet) as described in the paper. Additionally, this repository includes the following procedures: 
    
  1. Training the autoencoder.
  2. Loading a pre-trained autoencoder to train the `k-means` cluster in the ConvNet.    

### Setup the project environment

1. Clone or download the [repository](https://github.com/maheshkkumar/DLClustering).
2. Setup the `python 2.7` environment (either using virtualenv or anaconda distribution)
    1. **Virtualenv**
        1. Create a new environment: `virtualenv -p ~/path/python2.7 project_name`
        2. Activate the environment: `source ~/path_of_virtual_environment_folder/bin/activate`
        3. For more information: [Virtual Environments](https://docs.python-guide.org/dev/virtualenvs/)
    2. **Conda**
        1. Create a new environment: `conda create -n project_name python=2.7`
        2. Activate the environment: 
            1. `source ~/.bashrc`
            2. `source activate project_name`
        3. For more information: [Conda](https://conda.io/docs/user-guide/tasks/manage-environments.html)
4. Install all the project dependencies `pip --no-cache-dir install -r requirements.txt`

### Datasets

The following datasets were used for evaluating the proposed approach for deep latent feature unsupervised learning.

  1. [MNIST](http://yann.lecun.com/exdb/mnist/)
  2. [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
  
 >**Note**: The above mentioned datasets are included in the **Keras** framework. The hyperlinks can be visited to infer additional 
 information about the datasets

### Pre-trained models
The pre-trained **autoencoder** model weights are present in the cloned repository or they can be downloaded from the hyperlinks shown below: 
  
  1. **MNIST** -
        1. Without Attention: [Weights](https://myumanitoba-my.sharepoint.com/:u:/g/personal/kumarkm_myumanitoba_ca/EYg434t4_JxPiUL7902AznYB_T0D9khzz8Mt9IKmwBYZNQ?e=vCYcLD)
        2. With Attention: [Weights](https://myumanitoba-my.sharepoint.com/:u:/g/personal/kumarkm_myumanitoba_ca/EYg-iXOPRW5Fp1ELn8VeSwUBnLc6MZKC22tlUxbpk9snTw?e=7UdIlb)
  
  2. **Fashion-MNIST** - 
        1. Without Attention: [Weights](https://myumanitoba-my.sharepoint.com/:u:/g/personal/kumarkm_myumanitoba_ca/ESwX94u7VDhEiL1CpT9NQ4oB5sKXDx9rEyYScWhB6NiMEg?e=xFofTz)
        2. With Attention: [Weights](https://myumanitoba-my.sharepoint.com/:u:/g/personal/kumarkm_myumanitoba_ca/EUWjRulHGv5AgFAkd9p6aY8BP3G1q8JIbFxJ_s5yvqZi-Q?e=bbUleZ)

Or download the entire models folder: [Download](https://myumanitoba-my.sharepoint.com/:f:/g/personal/kumarkm_myumanitoba_ca/EiKq8S-DvrhOuywYlYwnUnYBXt-ofblSB7bXw8PzPpf4cg?e=zFY78p)

> **Note**: Place the downloaded weights in the **models** folder.

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
                                
### Using a pre-trained autoencoder to train the clustering layer

Use the downloaded pre-trained autoencoder to start the clustering task.

1. **MNIST**: 

    1. **Without Attention**: 
            ```
            python train.py -d mnist -m cae -aiter 500 -citer 20000 -aew ./models/mnist/mnist_ae_without_attention.h5 -lr 1
            ```

    2. **With Attention**: 
            ```
            python train.py -d mnist -m cae -aiter 500 -citer 20000 -aew ./models/mnist/mnist_ae_with_attention.h5 -lr 0.1 -att True
            ```
2. **Fashion-MNIST**: 

    1. **Without Attention**: 
            ```
            python train.py -d fmnist -m cae -aiter 500 -citer 20000 -aew ./models/fmnist/fmnist_ae_without_attention.h5 -lr 1
            ```
 
    2. **With Attention**: 
            ```
            python train.py -d fmnist -m cae -aiter 500 -citer 20000 -aew ./models/fmnist/fmnist_ae_with_attention.h5 -lr 0.1 -att True
            ```

The training script stops on either completing the specified clustering iterations or upon reaching the tolerance level. The code to generate the plots are in **notebooks** folder.

**Note**: The code for few concepts are borrowed from [DEC-Keras](https://github.com/XifengGuo/DEC-keras).
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

1. MNIST: ```python visualize_results.py -d mnist -m dae -aew ./models/mnist/mnist_ae_model.h5```
1. Fashion-MNIST: `python visualize_results.py -d fmnist -m dae -aew ./models/mnist_ae_model.h5`
-->
