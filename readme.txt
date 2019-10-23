# Project Title

Code for conference submission: Deep Neural Networks are Biased towards Simple Functions

## Getting Started

The code here can be used to recreate the raw data for some results shown in the expirements of our paper accepted to NIPS. Note, this code does not produce the actual plots, simply the raw data that can then be used to output plots. All code was run on Python using Keras and a Tensorflow backend. 

### Prerequisites

In addition to installing Tensorflow, various python packages will be needed for running this code. These packages include NumPy, Pandas, keras, and scipy. We recommend running the code using an Anaconda environment (https://www.anaconda.com/download/).

## Running the code

To recreate our results, the following scripts can be run to create csv files that can then be analyzed to re-create our results. Note, all scripts import utils.py or MNIST_utils.py, since these contain a set of functions that are used throughout the different scripts.

Descriptions of each script:
* greedy_search.py: this script simulates a greedy search on a large set of models to find nearest adversarial points in a binary input space (subsection 4.1). 
* hamming_distance_exact.py: this script simulates an exhaustive search on a large set of models to validate findings of greedy search (appendix). Before running this, the create_hamming_pickle.py file must be run to create a pickle that contains the list of points to exhaustively search at each hamming distance for a network size.
* random_bit_flips.py: this script simulates the hamming distance to different classification for random bit flips (subsection 4.2).
* MNIST_run_models.py: performs analysis on MNIST data to output raw data for figure 3 (subsection 4.3). Before running this script, models for classifying MNIST data need to be created, trained, and saved to the ../models directory. This can be performed with the  create_MNIST_models.py script.
* MNIST_run_models_boundary: performs analysis on MINIST data to output raw data for figure 4 (subsection 4.3). Before running this script, models for classifying MNIST data need to be created, trained, and saved to the ../models directory. This can be performed with the  create_MNIST_models.py script.

Note, scripts listed above have a name variable which can be used to name the outputted csv files (and pickle files where relevant). CSV files are by default stored in ../csv_files directory. Pickle files may also be created in some of these scripts. They are stored by default in the ../pickles directory. For MNIST analysis, models are trained and subsequently stored in the ../models directory. 