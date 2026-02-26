from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def build_gru_model(input_shape):
    model = Sequential()

    model.add(GRU(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(GRU(64))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def train_gru_model(model, X_train, y_train):
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        "models/best_gru.keras",
        monitor='val_loss',
        save_best_only=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    return history