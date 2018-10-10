import numpy as np
import pandas as pd
from keras.models import load_model

model_name = 'models/543-0.13.h5'
model = load_model(model_name)

def readJSON(json_file):
  data = pd.read_json(json_file)

  id_number = data['id'].values
  band1 = np.array([np.array(item, dtype=np.float32).reshape(75, 75) for item in data['band_1']])
  band2 = np.array([np.array(item, dtype=np.float32).reshape(75, 75) for item in data['band_2']])
  band3 = (band1 + band2) / 2.0
  X = np.concatenate((band1[:,:,:,np.newaxis], band2[:,:,:,np.newaxis], band3[:,:,:,np.newaxis]), axis=-1)
  return id_number, X

test_file = 'test.json'
id_number, x = readJSON(test_file)
y = model.predict(x)
y = [item[0] for item in y]

submission = pd.concat([pd.Series(id_number, name='id'), pd.Series(y, name='is_iceberg')], axis=1)
submission.to_csv('submission.csv', index=False, header=True)