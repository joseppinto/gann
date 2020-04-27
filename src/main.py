import pandas as pd
from nn import *
from ga import *
from sklearn.model_selection import StratifiedShuffleSplit

# Import datasets
train = pd.read_csv("../data/train.csv")
x = train.drop('cancer', axis=1).values
y = train['cancer'].values

sss = StratifiedShuffleSplit(n_splits=1)
for train_idx, valid_idx in sss.split(x, y):
    train_x, valid_x = x[train_idx], x[valid_idx]
    train_y, valid_y = y[train_idx], y[valid_idx]


# Run genetic algorithm
toolbox = init_ga(run_dna,                             # Pass evaluation function (buils and trains from dna)
                  train_x, train_y, valid_x, valid_y,  # Pass problem data
                  eval_epochs=10
                  )


run_ga('df.csv', toolbox, pop_size=50, max_gens=100)




