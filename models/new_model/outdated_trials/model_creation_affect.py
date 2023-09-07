import pandas as pd
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load the annotations for training and validation from separate CSV files
train_annotations_path = 'dataset/AffectNet/affect_train.csv'
valid_annotations_path = 'dataset/AffectNet/affect_val.csv'
train_annotations_df = pd.read_csv(train_annotations_path)
valid_annotations_df = pd.read_csv(valid_annotations_path)

image_size = (224, 224) 
batch_size = 64


datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = datagen.flow_from_dataframe(
    dataframe=train_annotations_df,
    directory="",
    x_col="image_path",
    y_col="label",  # If you have labels in your CSV
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)


valid_generator = datagen.flow_from_dataframe(
    dataframe=valid_annotations_df,
    directory="",
    x_col="image_path",
    y_col="label",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
)


# Load the EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a Global Average Pooling layer and a Dense layer for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(8, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10  # You can adjust the number of epochs based on your dataset and convergence
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    epochs=epochs
)

# Save the trained model
model.save('affectNet_emotion_model.h5')