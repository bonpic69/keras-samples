import numpy as np
np.random.seed(123) # random number generator replicable

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist

#load preshuffled data into train and test sets
(X_train,y_train),(X_test,y_test) = mnist.load_data()

print X_train.shape

from matplotlib import pyplot as plt
plt.imshow(X_train[0])
plt.show()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

print X_train.shape

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalize values between 0-1
X_train /= 255
X_test /= 255

print y_train.shape

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print Y_train.shape

#Create the model

model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))

print model.output_shape

model.add(Convolution2D(32,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.layers()

#compiling model

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Fit the Keras model to training data
history = model.fit(X_train, Y_train, validation_split=0.33, batch_size=32, 
          nb_epoch=2, verbose=1)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
          
#Evaluate Keras model
score = model.evaluate(X_test, Y_test, verbose=0)
