import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard

def buildModel(saved_model=None):
  optimizer = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)

  if saved_model is not None:
    model = load_model(saved_model)
  else:
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=(75, 75, 3)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

def readAndParseInput(json_file):
  data = pd.read_json(json_file)
  band1 = np.array([np.array(item, dtype=np.float32).reshape(75, 75) for item in data['band_1']])
  band2 = np.array([np.array(item, dtype=np.float32).reshape(75, 75) for item in data['band_2']])
  band3 = (band1 + band2) / 2.0
  X = np.concatenate((band1[:,:,:,np.newaxis], band2[:,:,:,np.newaxis], band3[:,:,:,np.newaxis]), axis=-1)
  Y = data['is_iceberg'].values.copy()
  return X, Y

train_file = 'train.json'
X, Y = readAndParseInput(train_file)

model = buildModel(saved_model='models/529-0.12.h5')
initial_epoch = 529

# Learning rate annealer & utility callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=30, verbose=1,
                                            factor=0.5, min_lr=1e-6)

model_check_point = ModelCheckpoint('models/{epoch:03d}-{val_loss:.2f}.h5',
                                    save_best_only=True, period=1)
tensorboard_visualization = TensorBoard(log_dir='logs/')

# Data augmentation
random_seed = 2
epochs = 10000
batch_size = 40
test_split_ratio = 0.05

datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                             samplewise_center=False,  # set each sample mean to 0
                             featurewise_std_normalization=False,  # divide inputs by std of the dataset
                             samplewise_std_normalization=False,  # divide each input by its std
                             zca_whitening=False,  # apply ZCA whitening
                             rotation_range=180, # randomly rotate images in the range (degrees, 0 to 180)
                             zoom_range=0.2,  # randomly zoom image
                             width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
                             height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
                             horizontal_flip=True,  # randomly flip images
                             vertical_flip=True,  # randomly flip images
                             rescale=None,
                             validation_split=test_split_ratio)
                             
train_generator = datagen.flow(X, Y,
                               batch_size=batch_size,
                               shuffle=True,
                               seed=random_seed,
                               subset='training')
validation_generator = datagen.flow(X, Y,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    seed=random_seed,
                                    subset='validation')

# Train the model
history = model.fit_generator(train_generator,
                              epochs=epochs,
                              steps_per_epoch=256,
                              validation_data=validation_generator,
                              verbose=1, 
                              callbacks=[learning_rate_reduction, model_check_point, tensorboard_visualization],
                              initial_epoch=initial_epoch)

model.save('models/sep28_manual.h5')