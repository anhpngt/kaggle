from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
import cv2

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
  # Load model
  model = load_model('models/jan9-033-0.17.h5')

  # Load test data
  test_dir = 'data/test'
  test_files = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
  file_number = len(test_files)
  del test_files

  # Get the images by batch and predict
  x = []
  results = np.zeros(file_number)
  for i in range(file_number):
    x.append(cv2.resize(cv2.imread(join(test_dir, str(i+1) + '.jpg')), (128, 128)))
    if len(x) == 1777 or i == file_number - 1:
      print('Calculating score at i = {}'.format(i))
      x = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in x], dtype=np.float32)
      x = x / 255.
      scores = model.predict(x)
      scores_sum_rowwise = np.sum(scores, axis=1)
      scores[:, 0] /= scores_sum_rowwise
      scores[:, 1] /= scores_sum_rowwise
      results[i-x.shape[0]+1:i+1] = scores[:,1]
      x = []

  submission = pd.concat([pd.Series(range(1,12501), name='id'), pd.Series(results, name='label')], axis=1)
  submission.to_csv("submission.csv", index=False)
