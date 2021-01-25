from keras.datasets import mnist
import keras
import tensorflow as tf
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn import metrics,pipeline,svm
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder



def NeuralNetwork(neurons):
    input=tf.keras.Input(shape=(28*28))
    hidden = tf.keras.layers.Dense(neurons, activation='relu')(input)
    output=tf.keras.layers.Dense(10, activation='linear')(hidden)
    NN = tf.keras.models.Model(inputs=input, outputs=output)
    return NN
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X=train_X.reshape(-1,28*28)/(28*28)
train_X=train_X[0:500,:]
train_y=train_y[0:500]
train_y_onehot = np.zeros((train_y.size, train_y.max()+1))
train_y_onehot[np.arange(train_y.size),train_y] = 1


test_X=test_X.reshape(-1,28*28)/(28*28)
test_X=test_X[0:1000,:]
test_y=test_y[0:1000]
test_y_onehot = np.zeros((test_y.size, test_y.max()+1))
test_y_onehot[np.arange(test_y.size),test_y] = 1

train_loss = []
test_loss = []

sizes=[10,50,100,200]
loss_fn = tf.keras.losses.mean_squared_error
for neurons in sizes:
    model=NeuralNetwork(neurons)
    opt = tf.keras.optimizers.SGD(lr=0.1)
    loss_fn = tf.keras.losses.mean_squared_error
    model.compile(loss=loss_fn, optimizer=opt)
    model.fit(train_X, train_y_onehot, batch_size=100, epochs=5000)
    train_loss.append(model.evaluate(train_X, train_y_onehot))
    test_loss.append(model.evaluate(test_X, test_y_onehot))
plt.plot(sizes,train_loss,label='Training')
plt.plot(sizes,test_loss,label='Testing')
plt.xlabel('neurons')
plt.ylabel('l2 loss')
plt.grid()
plt.legend()
plt.savefig('double_descent_NN.png', dpi=150)