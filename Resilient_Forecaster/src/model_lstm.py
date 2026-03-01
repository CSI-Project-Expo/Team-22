# Import os module to work with folders and file system
import os

# Import numpy library for numerical operations and arrays
import numpy as np

# Import Sequential model (used to create neural network layer by layer)
from tensorflow.keras.models import Sequential

# Import LSTM and Dense layers
# LSTM = Long Short-Term Memory (used for time series prediction)
# Dense = Fully connected layer
from tensorflow.keras.layers import LSTM, Dense

# Import EarlyStopping to stop training when model stops improving
from tensorflow.keras.callbacks import EarlyStopping


# Import additional layers
# Dropout = prevents overfitting by randomly disabling some neurons
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Import Adam optimizer (used to improve learning)
from tensorflow.keras.optimizers import Adam


# Function to build (create) the LSTM model
# input_shape = shape of training data (time steps, features)
def build_model(input_shape):

    # Create a Sequential model (layers added one by one)
    model = Sequential()

    # Add first LSTM layer with 64 neurons
    # return_sequences=True means output goes to next LSTM layer
    # input_shape defines input data shape
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))

    # Add Dropout layer to reduce overfitting (20% neurons ignored randomly)
    model.add(Dropout(0.2))

    # Add second LSTM layer with 64 neurons
    model.add(LSTM(64))

    # Add another Dropout layer
    model.add(Dropout(0.2))

    # Add Dense layer with 32 neurons and ReLU activation
    # ReLU helps model learn complex patterns
    model.add(Dense(32, activation='relu'))

    # Add final Dense layer with 1 neuron
    # This gives the final predicted output (stock price)
    model.add(Dense(1))

    # Create Adam optimizer with learning rate 0.001
    optimizer = Adam(learning_rate=0.001)

    # Compile the model
    # optimizer = Adam (learning method)
    # loss = mean squared error (used for prediction problems)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Return the built model
    return model


# Import callbacks
# EarlyStopping = stops training early if no improvement
# ModelCheckpoint = saves best model automatically
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Function to train the model
# model = created LSTM model
# X_train = input training data
# y_train = output training data
def train_model(model, X_train, y_train):
    
    # Create EarlyStopping object
    # monitor='val_loss' means watch validation loss
    # patience=5 means stop if no improvement for 5 epochs
    # restore_best_weights=True restores best model weights
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Create ModelCheckpoint object
    # Saves best model in "models" folder
    # save_best_only=True saves only best model
    checkpoint = ModelCheckpoint(
        "models/best_lstm.keras",
        monitor='val_loss',
        save_best_only=True
    )

    # Train the model using training data
    history = model.fit(

        # Input training data
        X_train,

        # Output training data
        y_train,

        # epochs=40 means training runs 40 times max
        epochs=40,

        # batch_size=32 means 32 samples processed at once
        batch_size=32,

        # validation_split=0.2 means 20% data used for validation
        validation_split=0.2,

        # callbacks used during training
        callbacks=[early_stop, checkpoint],

        # verbose=1 shows training progress
        verbose=1
    )

    # Return training history (loss, validation loss, etc.)
    return history