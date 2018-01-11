import numpy as np
import pandas as pd

np.random.seed(2)

from sklearn.model_selection import train_test_split
import itertools

from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import load_model

if __name__ == '__main__':
  random_seed = 2

  # CNN model

  # Resume training model
  resuming_epoch = 33
  model = load_model('models/jan9-033-0.17.h5')

  # Create new model
  # resuming_epoch = 0
  # model = Sequential()
  # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)))
  # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
  # model.add(MaxPool2D(pool_size=(2, 2)))
  # model.add(BatchNormalization())
  # # model.add(Dropout(0.25))

  # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
  # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
  # model.add(MaxPool2D(pool_size=(2, 2)))
  # model.add(BatchNormalization())
  # model.add(Dropout(0.25))

  # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
  # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
  # model.add(MaxPool2D(pool_size=(2, 2)))
  # model.add(BatchNormalization())
  # # model.add(Dropout(0.25))

  # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
  # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
  # model.add(MaxPool2D(pool_size=(2, 2)))
  # model.add(BatchNormalization())
  # model.add(Dropout(0.25))

  # model.add(Flatten())
  # model.add(Dense(256, activation='relu'))
  # model.add(Dropout(0.25))
  # model.add(Dense(256, activation='relu'))
  # model.add(Dropout(0.25))
  # model.add(Dense(2, activation='softmax'))

  # Optimizer
  # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)
  optimizer = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

  # Compile the model
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  # Set a learning rate annealer
  learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=6, verbose=1,
                                              factor=0.5, min_lr=1e-5)

  # Additional utility callbacks
  model_check_point = ModelCheckpoint('models/jan9-{epoch:03d}-{val_loss:.2f}.h5',
                                      save_best_only=True, period=1)
  tensorboard_visualization = TensorBoard(log_dir='logs/')

  # Data augmentation
  epochs = 200
  batch_size = 60
  # steps_per_epoch = 
  datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                               samplewise_center=False,  # set each sample mean to 0
                               featurewise_std_normalization=False,  # divide inputs by std of the dataset
                               samplewise_std_normalization=False,  # divide each input by its std
                               zca_whitening=False,  # apply ZCA whitening
                               # randomly rotate images in the range (degrees, 0 to 180)
                               rotation_range=180,
                               zoom_range=0.2,  # Randomly zoom image
                               # randomly shift images horizontally (fraction of total width)
                               width_shift_range=0.2,
                               # randomly shift images vertically (fraction of total height)
                               height_shift_range=0.2,
                               horizontal_flip=True,  # randomly flip images
                               vertical_flip=True,  # randomly flip images
                               rescale=1. / 255)
  train_generator = datagen.flow_from_directory('data/train',
                                                target_size=(128, 128),
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                seed=random_seed)
  validation_generator = datagen.flow_from_directory('data/validation',
                                                     target_size=(128, 128),
                                                     batch_size=batch_size,
                                                     class_mode='categorical',
                                                     seed=random_seed)

  # Train the model
  history = model.fit_generator(train_generator,
                                epochs=epochs,
                                steps_per_epoch=len(train_generator),
                                validation_data=validation_generator,
                                verbose=1,
                                callbacks=[learning_rate_reduction, 
                                           model_check_point, 
                                           tensorboard_visualization],
                                initial_epoch=resuming_epoch)

  model.save('models/jan10.h5')
