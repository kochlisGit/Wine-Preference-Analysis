from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow_addons.layers import GELU
from tensorflow_addons.optimizers import Yogi
import tensorflow as tf


def build_model(input_shape, hidden_units, show_summary=True):
    model = Sequential()

    model.add(Input(shape=input_shape))
    hidden_layers = len(hidden_units)
    for units in hidden_units[0:hidden_layers-1]:
        model.add(Dense(units=units))
        model.add(BatchNormalization())
        model.add(GELU())
        model.add(Dropout(rate=0.2))

    model.add(Dense(units=hidden_layers-1, kernel_regularizer='l1'))
    model.add(BatchNormalization())
    model.add(GELU())
    model.add(Dropout(rate=0.2))

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=Yogi(learning_rate=0.001), loss='huber')

    if show_summary:
        model.summary()

    return model


def train(model, name, x_train, y_train, batch_size, epochs):
    checkpoint = ModelCheckpoint(
        filepath='weights/' + name + '_callback.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=25,
        verbose=1,
        restore_best_weights=True
    )

    callbacks = [checkpoint, early_stopping]

    kfold = KFold(n_splits=10)

    for e in range(epochs):
        print('-- Epoch --', e+1)

        for train_index, test_index in kfold.split(x_train):
            train_inputs = x_train[train_index]
            train_targets = y_train[train_index]
            test_inputs = x_train[test_index]
            test_targets = y_train[test_index]

            model.fit(
                x=train_inputs,
                y=train_targets,
                batch_size=batch_size,
                epochs=1,
                verbose=1,
                callbacks=callbacks,
                validation_data=(test_inputs, test_targets)
            )


def load_model(checkpoint_path):
    return tf.keras.models.load_model(checkpoint_path)


def predict(model, x_test):
    return model.predict(x_test)
