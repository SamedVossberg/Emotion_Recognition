from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Activation
from keras.initializers import VarianceScaling, Zeros, Ones
from keras.regularizers import Regularizer
from keras.constraints import Constraint
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from kerastuner.tuners import RandomSearch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# TODO: use hp to tune the model
# Model initailization
def build_model(hp):
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='linear', input_shape=(48, 48, 1), name="conv2d_1"))
    model.add(Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='linear', name="conv2d_2"))
    model.add(BatchNormalization(name="batch_normalization_1"))
    model.add(Activation(activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 1), padding='same', activation='linear', name="conv2d_3"))
    model.add(Conv2D(filters=128, kernel_size=(1, 3), padding='same', activation='linear', name="conv2d_4"))
    model.add(BatchNormalization(name="batch_normalization_2"))
    model.add(Activation(activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=256, kernel_size=(3, 1), padding='same', activation='linear', name="conv2d_5"))
    model.add(Conv2D(filters=256, kernel_size=(1, 3), padding='same', activation='linear', name="conv2d_6"))
    model.add(BatchNormalization(name="batch_normalization_3"))
    model.add(Activation(activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=512, kernel_size=(3, 1), padding='same', activation='linear', name="conv2d_7"))
    model.add(Conv2D(filters=512, kernel_size=(1, 3), padding='same', activation='linear', name="conv2d_8"))
    model.add(BatchNormalization(name="batch_normalization_4"))
    model.add(Activation(activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='linear', name="dense_1"))
    model.add(BatchNormalization(name="batch_normalization_5"))
    model.add(Activation(activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax', name="dense_2"))

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])



# Load the dataset
data = pd.read_csv('fer2013.csv')

# Preprocess the data
pixels = data['pixels'].apply(lambda x: np.fromstring(x, sep=' ')).values
pixels = np.vstack(pixels) / 255.0  # normalize pixel values
pixels = pixels.reshape(-1, 48, 48, 1)  # reshape for the model

emotions = to_categorical(data['emotion'])  # one-hot encoding

# Split the data into training, validation, and test sets
train_pixels, test_pixels, train_emotions, test_emotions = train_test_split(pixels, emotions, test_size=0.1, random_state=42)
train_pixels, val_pixels, train_emotions, val_emotions = train_test_split(train_pixels, train_emotions, test_size=0.1, random_state=41)

model = build_model()

# Use ModelCheckpoint to save the best model during training
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# Train the model
history = model.fit(train_pixels, train_emotions,
                    validation_data=(val_pixels, val_emotions),
                    epochs=50, batch_size=64, callbacks=[checkpoint])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_pixels, test_emotions)
print('Test accuracy:', test_acc)



tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='output',
    project_name="Emotion_Detection")

tuner.search(train_pixels, train_emotions,
             epochs=3,
             validation_data=(val_pixels, val_emotions))