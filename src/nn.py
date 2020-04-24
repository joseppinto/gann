import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from ga import DENSE_ACTIVATIONS

session_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=session_config)


# Translate dna defined in string to model
def model_from_dna(dna):
    batch_size = int(dna[0])            # First gene -> batch size
    lr = float(dna[1])                  # Second gene -> learning rate

    model = Sequential()
    for i in range(2, dna.size):        # Remaining genes are neuron layers
        if dna[i] < 0:
            model.add(BatchNormalization())
        elif dna[i] > 1:                # If it is a dense layer, integer part is number of nodes, decimal is activation
            model.add(Dense(int(dna[i]), activation=DENSE_ACTIVATIONS[round((dna[i] % 1) * 10) / 10.0]))
        elif dna[i] < 1:                # No integer part means dropout layer
            model.add(Dropout(dna[i]))

    model.add(Dense(2, activation='sigmoid'))   # Final layer will always be the same (binary classification)

    return batch_size, lr, model                # Return batch size, learning rate, and built model


# Build, train and test neural network architecture from dna string
# Returns average of 5 best accuracies reached in training test
# This function evaluates the performance of a nn architecture on a given dataset
def run_dna(dna, eval_epochs, train_x, train_y, valid_x, valid_y):
    batch_size, lr, model = model_from_dna(dna)             # Extract batch size and learning rate and build model

    opt = Adam(lr=lr)                                       # Configure optimizer with learning rate
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=eval_epochs,
                        verbose=1,
                        validation_data=(valid_x, valid_y)
                        )
    accuracy = list(history.history['val_accuracy'])                # Get accuracies from train history
    accuracy.sort(reverse=True)
    best = accuracy[:3]
    return sum(best)/len(best)
