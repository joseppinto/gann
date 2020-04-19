import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

session_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=session_config)


# Translate dna defined in string to model
def model_from_dna(dna):
    arr = dna.split(",")                # Genes separated by commas ','
    batch_size = int(arr[0])            # First gene -> batch size
    lr = float(arr[1])                  # Second gene -> learning rate

    model = Sequential()
    for i in range(2, len(arr)):        # Remaining genes are neuron layers
        type = arr[i][0]                # First symbol -> type of layer
        opt = arr[i][1]                 # Second symbol -> option relevant to layer type (e.g. activation function)
        if type == 'd':
            num = float(arr[i][2:])     # Remaining symbols, layer structure properties (e.g. number of neurons)
            model.add(Dropout(num))
        elif type == 'D':
            num = int(arr[i][2:])
            if opt == 'r':
                act = 'relu'
            elif opt == 't':
                act = 'tanh'
            elif opt == 'l':
                act = 'linear'
            elif opt == 's':
                act = 'sigmoid'
            model.add(Dense(num, activation=act))
        elif type == 'N':
            model.add(BatchNormalization())

    model.add(Dense(2, activation='sigmoid'))   # Final layer will always be the same (binary classification)

    return batch_size, lr, model                # Return batch size, learning rate, and built model


# Build, train and test neural network architecture from dna string
# Returns average of 5 best accuracies reached in training test
# This function evaluates the performance of a nn architecture on a given dataset
def run_dna(dna, epochs, train_x, train_y, valid_x, valid_y):
    batch_size, lr, model = model_from_dna(dna)             # Extract batch size and learning rate and build model

    opt = Adam(lr=lr)                                       # Configure optimizer with learning rate
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(valid_x, valid_y)
                        )
    accs = list(history.history['val_accuracy'])                # Get accuracies from train history
    accs.sort(reverse=True)
    return sum(accs[:5])/5
