import numpy as np
import keras
import tensorflow as tf
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn import metrics,pipeline,svm
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def NeuralNetwork(neurons):
    input=tf.keras.Input(shape=(1))
    hidden = tf.keras.layers.Dense(neurons, activation='relu')(input)
    output=tf.keras.layers.Dense(1, activation='linear')(hidden)
    NN = tf.keras.models.Model(inputs=input, outputs=output)
    return NN

n=50
train_X=np.random.rand(n)*0.5+0.5
train_X=np.expand_dims(train_X,axis=1)
train_y=np.sqrt(train_X*(1-train_X))*np.sin(2.1*np.pi/(train_X+0.05))

n=1000
test_X=np.random.rand(n)*0.5+0.5
test_X=np.expand_dims(test_X,axis=1)
test_y=np.sqrt(test_X*(1-test_X))*np.sin(2.1*np.pi/(test_X+0.05))

train_loss = []
test_loss = []

sizes=[10000]
loss_fn = tf.keras.losses.mean_squared_error
for neurons in sizes:
    model = NeuralNetwork(neurons)
    opt = tf.keras.optimizers.SGD(lr=0.1)
    loss_fn = tf.keras.losses.mean_squared_error
    model.compile(loss=loss_fn, optimizer=opt)
    model.fit(train_X, train_y, batch_size=25, epochs=5000)
    X_plot = np.linspace(0.5, 1, 10000)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot)
    plt.scatter(train_X, train_y)
    plt.show()
    train_loss.append(model.evaluate(train_X, train_y))
    test_loss.append(model.evaluate(test_X, test_y))
plt.plot(sizes,train_loss,label='Training loss')
plt.plot(sizes,test_loss,label='Testing loss')
plt.xlabel('n° RFF')
plt.ylabel('L2 loss')
plt.grid()
plt.show()