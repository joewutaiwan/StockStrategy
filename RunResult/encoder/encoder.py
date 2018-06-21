from keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
import matplotlib.cm as cm
import imp
fin_data = imp.load_source('fin_data', 'Library/fin_data.py')

(x_data, y_data, z_data) = fin_data.load_company_data()

encoder = load_model('RunResult/encoder/encoder.h5')
encoder.summary()

predict_x = x_data
predict_y = y_data.reshape(y_data.shape[0]) # to 1-d
predict_z = z_data.reshape(z_data.shape[0]) # to 1-d
result = encoder.predict(predict_x)

print predict_z
predict_z = 2**(predict_z - 3)

f, ax = plt.subplots(1)
for i in np.unique(predict_y):
    mask = predict_y == i
    plt.scatter(result[mask, 0], result[mask, 1], label=i, s=predict_z[mask])

ax.legend()
plt.show()
