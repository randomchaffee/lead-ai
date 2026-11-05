import numpy as np
from feature_quantifying import prepare_dataset

# set random seed for reproducibility
np.random.seed(42)

# load preprocessed dataset
X, rows = prepare_dataset() # X is 1600 by 5 normalized numpy matrix, rows is list of raw dicts


# start working dito, or pwedeng ilagay sa isang main function yung X, rows depende sa preference