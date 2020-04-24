import pandas as pd
from nn import *
from ga import *
import seaborn as sns
import matplotlib.pyplot as plt

# Import datasets
train = pd.read_csv("../data/train.csv")
train_x = train.drop('cancer', axis=1).values
train_y = train['cancer'].values

valid = pd.read_csv("../data/test.csv")
valid_x = valid.drop('cancer', axis=1).values
valid_y = valid['cancer'].values


# Run genetic algorithm
toolbox = init_ga(run_dna,                             # Pass evaluation function (buils and trains from dna)
                  train_x, train_y, valid_x, valid_y   # Pass problem data
                  )

for df in run_ga(toolbox, pop_size=10, max_gens=10):
    df.to_csv('df.csv', index=False)




