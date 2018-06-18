import numpy as np
import os
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.optimizers import RMSprop
from keras.models import Model
from datetime import datetime
import matplotlib.pyplot as plt
import imp
fin_data = imp.load_source('fin_data', 'Library/fin_data.py')
version = imp.load_source('version', 'Library/version.py')
save_weights_path = 'FinancialStatement/autoencoder/autoencoder_weights.h5'
save_encoder_path = 'RunResult/encoder'
result_name = "encoder.h5"

(x_train, y_train), (x_test, y_test) = fin_data.load_data(0.99)

# in order to plot in a 2D figure
encoding_dim = 2
x_dim = 64

# this is our input placeholder
input_img = Input(shape=(x_dim,))

# encoder layers

x = Dense(x_dim, activation='relu')(input_img)
encoded = Dense(512, activation='relu')(x)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
y = Dense(encoding_dim, activation='relu')(encoded)

# decoder layers
decoded = Dense(64, activation='relu')(y)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
z = Dense(x_dim, activation='sigmoid')(decoded)

# construct the autoencoder model
autoencoder = Model(inputs=input_img, outputs=z)
try:
    autoencoder.load_weights(save_weights_path)
except IOError:
    print 'autoencoder_weights.h5 is no found'
    pass
    
# construct the encoder model for plotting
encoder = Model(inputs=input_img, outputs=y)

# compile autoencoder
autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')

autoencoder.summary()

# training
autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()

time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print "finished at :" + time_str
version.save(
    result_model = encoder, 
    save_encoder_path = save_encoder_path,
    result_name = result_name,
    time_str = time_str
)
autoencoder.save_weights(save_weights_path)

