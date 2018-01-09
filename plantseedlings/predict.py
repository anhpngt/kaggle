from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
import cv2

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
  # Load test data
  test_dir = 'data/test'
  test_files = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
  x = [cv2.resize(cv2.imread(join(test_dir, f)), (100, 100)) for f in test_files]
  x = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in x], dtype=np.float32)
  x = x / 255.

  classes = ['Black-grass',
             'Charlock',
             'Cleavers',
             'Common Chickweed',
             'Common wheat',
             'Fat Hen',
             'Loose Silky-bent',
             'Maize',
             'Scentless Mayweed',
             'Shepherds Purse',
             'Small-flowered Cranesbill',
             'Sugar beet']

  # Load model
  model = load_model('models/6pm-jan7-114-0.01.h5')

  # Predict
  scores = model.predict(x)
  results = np.argmax(scores, axis=1)
  results = [classes[prediction] for prediction in results]
  submission = pd.concat([pd.Series(test_files, name='file'), pd.Series(results, name='species')], axis=1)
  submission.to_csv("submission.csv", index=False)
