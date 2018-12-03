## Introduction

A method to employ a neural network to learn the latent representation of the multimedia data (e.g., images) to better form the clusters in a lower dimensional space.

### Steps to setup the project

1. Clone the repository.
2. Setup the `python 2.7` environment (either conda or pip)
    1. Via `virtualenv`
        1. Create a new environment: `virtualenv -p ~/path/python2.7 project_name`
    2. Via `conda`
        1. Create a new environment: `conda create -n python=2.7 project_name`
3. Activate the environment: `source project_name/bin/activate`
4. Install all the dependencies `pip install -r requirements.txt`
5. Run `train.py -h` to know the additional command line parameters.
6.     usage: train.py [-h] [-il INCLUDE_LAYER] [-d {mnist,fmnist,stl,cifar10}]
                [-bs BATCH_SIZE] [-att ATTENTION] [-m {ae,dae}]
                [-citer CLUSTER_ITERATIONS] [-aiter AE_ITERATIONS]
                [-iu INTERVAL_UPDATION] [-tt TOLERANCE_THRESHOLD]
                [-aew AE_WEIGHTS] [-od OUTPUT_DIRECTORY]

        optional arguments:
          -h, --help            show this help message and exit
          -il INCLUDE_LAYER, --include_layer INCLUDE_LAYER
                                Include an additional layer in auto encoder
          -d {mnist,fmnist,stl,cifar10}, --dataset {mnist,fmnist,stl,cifar10}
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
                                Weights of pre-trained auto-encoder
          -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                                Path of the output directory to store the results and
                                training models
7. Demo run: `python train.py -d mnist -att True -aiter 10 -citer 10 -aew ./ae_weights/model.h5`
8. Generate result graphs: `python generate_results.py` 