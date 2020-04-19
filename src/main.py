import tensorflow as tf
import pandas as pd
from .nn import *
from .ga import *

# Import datasets
train = pd.read_csv("../data/train.csv")
train_x = train.drop('cancer', axis=1).values
train_y = train['cancer'].values

valid = pd.read_csv("../data/test.csv")
valid_x = valid.drop('cancer', axis=1).values
valid_y = valid['cancer'].values


# Run genetic algorithm
run_ga(run_dna,                             # Pass evaluation function (builds, trains and evaluates models from dna)
       train_x, train_y, valid_x, valid_y   # Pass problem data
       )



