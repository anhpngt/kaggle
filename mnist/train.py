# https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd

np.random.seed(2)

from sklearn.model_selection import train_test_split
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import load_model

if __name__ == '__main__':
  # Data input
  train_df = pd.read_csv('/home/echo/mnist/train.csv')
  Y_train = train_df['label']
  X_train = train_df.drop('label', axis=1)
  del train_df

  # Normalization
  X_train = X_train / 255.0

  # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
  X_train = X_train.values.reshape(-1,28,28,1)

  # Label encoding to one-hot vector
  Y_train = to_categorical(Y_train, num_classes=10)

  # Splitting training and validation set
  random_seed = 2
  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 
                                                    test_size=0.1, random_state=random_seed)

  # plt.imshow(X_train[0][:,:,0])
  # print('Label: {}'.format(Y_train[0]))
  # plt.show()

  # CNN model
  # Resume training model
  # model = load_model('models/mnist_cnn_bn_regu.h5')
  # resuming_epoch = 30
  # Create new model
  model = Sequential()
  model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', kernel_regularizer=l2(0.01), input_shape=(28,28,1)))
  model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', kernel_regularizer=l2(0.01)))
  model.add(MaxPool2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Dropout(0.25))

  model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu', kernel_regularizer=l2(0.01)))
  model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu', kernel_regularizer=l2(0.01)))
  model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
  model.add(BatchNormalization())
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
  model.add(BatchNormalization())
  model.add(Dropout(0.25))
  model.add(Dense(10, activation='softmax', kernel_regularizer=l2(0.01)))

  # Optimizer
  # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)
  optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

  # Complile the model

  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  # Set a learning rate annealer
  learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 
                                              factor=0.5, min_lr=1e-5)

  # Additional utility callbacks
  model_check_point = ModelCheckpoint('models/cnn_bn_regu-{epoch:03d}-{val_loss:.2f}.h5',
                                      save_best_only=True, period=5)
  tensorboard_visualization = TensorBoard(log_dir='logs/', histogram_freq=10)

  # Data augmentation to prevent overfitting
  datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                               samplewise_center=False,  # set each sample mean to 0
                               featurewise_std_normalization=False,  # divide inputs by std of the dataset
                               samplewise_std_normalization=False,  # divide each input by its std
                               zca_whitening=False,  # apply ZCA whitening
                               rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                               zoom_range = 0.1, # Randomly zoom image 
                               width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                               height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                               horizontal_flip=False,  # randomly flip images
                               vertical_flip=False)  # randomly flip images
  datagen.fit(X_train)

  # Train the model
  epochs = 100 # Turn epochs to 30 to get 0.9967 accuracy
  batch_size = 86
  steps_per_epoch = int(X_train.shape[0] / batch_size)
  history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                                epochs=epochs, validation_data=(X_val, Y_val),
                                verbose=1, steps_per_epoch=steps_per_epoch,
                                callbacks=[learning_rate_reduction, model_check_point, tensorboard_visualization],
                                initial_epoch=resuming_epoch)

  model.save('models/mnist_cnn_bn_regu.h5')

  # Plot the loss and accuracy curves for training and validation 
  # fig, ax = plt.subplots(2,1)
  # ax[0].plot(history.history['loss'], color='b', label='Training loss')
  # ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
  # legend = ax[0].legend(loc='best', shadow=True)

  # ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
  # ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
  # legend = ax[1].legend(loc='best', shadow=True)
  # plt.show()