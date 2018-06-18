import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.optimizers import RMSprop

import fin_data

(x_train, y_train), (x_test, y_test) = fin_data.load_data(0.75)

print y_train.shape, y_train[0]
y_Train_OneHot = np_utils.to_categorical(y_train)
y_Test_OneHot = np_utils.to_categorical(y_test)
print(x_train.shape, x_test.shape)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=32))
model.add(Dropout(0.2))
for i in range(0, 20):
    model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.)

model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
train_history = model.fit(x_train, y_Train_OneHot, validation_split=0.2, 
                        epochs=10, batch_size=2000,verbose=2)

import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History 2')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

#show_train_history(train_history,'acc','val_acc')
#show_train_history(train_history,'loss','val_loss')
print(train_history.history["loss"])

print('Testing ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(x_test, y_Test_OneHot)

print('test loss: ', loss)
print('test accuracy: ', accuracy)


