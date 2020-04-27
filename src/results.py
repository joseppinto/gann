import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from ga import DENSE_ACTIVATIONS
import pandas as pd
import numpy as np

# Import datasets
train = pd.read_csv("../data/train.csv")
train_x = train.drop('cancer', axis=1).values
train_y = train['cancer'].values

test = pd.read_csv("../data/train.csv")
test_x = test.drop('cancer', axis=1).values
test_y = test['cancer'].values

session_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=session_config)


def model_from_dna(name, dna):
    batch_size = int(max(dna[0], 1))
    threshold = dna[0] - int(dna[0])
    lr = float(dna[1])

    model = Sequential()
    for i in range(2, dna.size):
        if dna[i] < 0:
            model.add(BatchNormalization())
        elif dna[i] >= 1:
            act = round(dna[i] - int(dna[i]), 1)
            if act > 0.4:
                act = 0.4
            elif act < 0.1:
                act = 0.1
            model.add(Dense(int(dna[i]), activation=DENSE_ACTIVATIONS[act]))
        elif dna[i] < 1:
            model.add(Dropout(dna[i]))

    model.add(Dense(1, activation='sigmoid'))

    return name, batch_size, threshold, lr, model


def baseline_model():
    lr = 0.001
    batch_size = 32
    threshold = 0.5

    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return 'baseline', batch_size, threshold, lr, model


def run_model(m):
    name, batch_size, threshold, lr, model = m
    opt = Adam(lr=lr)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=[tf.keras.metrics.Recall(name='recall', thresholds=threshold),
                           'accuracy',
                           tf.keras.metrics.AUC(name='auc')
                           ]
                  )
    tensorboard = TensorBoard(log_dir=f'logs/{name}')
    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=50,
              verbose=1,
              validation_data=(test_x, test_y),
              callbacks=[tensorboard]
              )

    tf.keras.utils.plot_model(model,
                              to_file=f'{name}.png',
                              show_shapes=True,
                              show_layer_names=True,
                              rankdir='TB',
                              )


def run_all():
    names = ['recall', 'auc', 'fitness']

    df = pd.read_csv('df.csv')
    df.drop_duplicates(subset=['architecture'], keep='first', inplace=True)

    models = [baseline_model()]
    for name in names:
        df2 = df.sort_values([name], ascending=False)
        models.append(model_from_dna(f"{name}(gen_{df2['gen'].values[0]})",
                                     np.array(eval(df2['architecture'].values[0]), dtype='float16')
                                     )
                      )

    for model in models:
        run_model(model)


run_all()
