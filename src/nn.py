import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from ga import DENSE_ACTIVATIONS

session_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=session_config)


# Translate dna defined in string to model
def model_from_dna(dna):
    batch_size = int(max(dna[0], 1))            # Integer of first gene -> batch size, don't allow null or negatives
    threshold = dna[0] - int(dna[0])            # Decimal of first gene -> classification threshold
    lr = float(dna[1])                          # Second gene -> learning rate

    model = Sequential()
    for i in range(2, dna.size):        # Remaining genes are neuron layers
        if dna[i] < 0:
            model.add(BatchNormalization())
        elif dna[i] >= 1:               # If it is a dense layer, integer part is number of nodes, decimal is activatio
            act = round(dna[i] - int(dna[i]), 1)
            # Fix a better workaround in the future
            if act > 0.4:
                act = 0.4
            elif act < 0.1:
                act = 0.1
            model.add(Dense(int(dna[i]), activation=DENSE_ACTIVATIONS[act]))
        elif dna[i] < 1:                # No integer part means dropout layer
            model.add(Dropout(dna[i]))

    model.add(Dense(1, activation='sigmoid'))   # Final layer will always be the same (binary classification)

    return batch_size, threshold, lr, model                # Return batch size, learning rate, and built model


# Build, train and test neural network architecture from dna string
# Returns average of 5 best accuracies reached in training test
# This function evaluates the performance of a nn architecture on a given dataset
def run_dna(dna, eval_epochs, train_x, train_y, valid_x, valid_y):
    batch_size, threshold, lr, model = model_from_dna(dna)  # Extract parameters and model

    opt = Adam(lr=lr)                                       # Configure optimizer with learning rate
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=[tf.keras.metrics.Recall(name='recall', thresholds=threshold),
                           'accuracy',
                           tf.keras.metrics.AUC(name='auc')
                           ]
                  )

    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=eval_epochs,
                        verbose=1,
                        validation_data=(valid_x, valid_y)
                        )

    recall = list(history.history['val_recall'])                                        # Get metrics from train history
    auc = list(history.history['val_auc'])
    metrics = sorted(zip(recall, auc), key=lambda t: t[1], reverse=True)[:3]   # Sort by recall
    # Return average of 3 best recalls (and average of respective accuracies)
    return sum([x[0] for x in metrics]) / len(metrics), \
        sum([x[1] for x in metrics]) / len(metrics),

