from keras.models import load_model, Model
import matplotlib.pyplot as plt
from keras.layers import Input
import imp
fin_data = imp.load_source('fin_data', 'Library/fin_data.py')

(x_train, y_train), (x_test, y_test) = fin_data.load_data(0.99)

encoder = load_model('RunResult/encoder/encoder.h5')
encoder.summary()

bengin = 5500
end = 5550
predict_x = x_train[bengin:end]
predict_y = y_train[bengin:end]
#predict_x = x_test
#predict_y = y_test

result = encoder.predict(predict_x)
plt.scatter(result[:, 0], result[:, 1], c=predict_y)
plt.colorbar()
plt.show()
