import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

from mainTrain import INPUT_SIZE

model=load_model('BrainTumor10EpochsCategorical.h5')

image = cv2.imread('pred\pred0.jpg')
img=Image.fromarray(image)

img=img.resize((INPUT_SIZE, INPUT_SIZE))

img=np.array(img)

input_image = np.expand_dims(img, axis=0)

prediction = model.predict(input_image)
result = np.argmax(prediction, axis=1)

print(result)


