from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Activation
from keras.initializers import VarianceScaling, Zeros, Ones
from keras.regularizers import Regularizer
from keras.constraints import Constraint

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
