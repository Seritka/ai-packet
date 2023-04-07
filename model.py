from tensorflow import keras
import numpy as np

np.random.seed(0)

model = keras.models.load_model('Chlee')
l = model.predict(np.array([5, 38, 3939, 27389]))

print(l)