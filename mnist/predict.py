import pandas as pd
import numpy as np

from keras.models import load_model

if __name__=='__main__':
  # Load test data
  test_df = pd.read_csv('test.csv')
  X_test = test_df.values.reshape(-1,28,28,1)
  del test_df
  X_test = X_test / 255.0

  # Load trained model
  model = load_model('models/mnist_cnn_bn_regu.h5')

  # Predict
  results = model.predict(X_test)
  results = np.argmax(results, axis=1)
  results = pd.Series(results, name='Label')
  submission = pd.concat([pd.Series(range(1,28001), name='ImageId'), results], axis=1)
  submission.to_csv("submission.csv", index=False)