import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

# Load model architecture from JSON file
with open("./models/model.json", "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load weights into the model
loaded_model.load_weights("./models/weights.h5")


# predictions = loaded_model.predict(xy)

