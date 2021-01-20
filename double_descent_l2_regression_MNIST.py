
from keras.datasets import mnist
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn import metrics,pipeline,svm
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X=train_X.reshape(-1,28*28)/(28*28)
train_y_onehot = np.zeros((train_y.size, train_y.max()+1))
train_y_onehot[np.arange(train_y.size),train_y] = 1

test_X=test_X.reshape(-1,28*28)/(28*28)
test_y_onehot = np.zeros((test_y.size, test_y.max()+1))
test_y_onehot[np.arange(test_y.size),test_y] = 1

train_acc = []
test_acc = []

feature_map_fourier = RBFSampler(gamma=.2, random_state=1)
fourier_approx_linear = pipeline.Pipeline([("feature_map", feature_map_fourier), ("svm",LinearRegression())])

features=[1000,1100,1200,1500]
for D in features:
    fourier_approx_linear.set_params(feature_map__n_components=D)
    fourier_approx_linear.fit(train_X, train_y_onehot)
    train_acc.append(fourier_approx_linear.score(train_X, train_y_onehot))
    test_acc.append(fourier_approx_linear.score(test_X, test_y_onehot))
plt.plot(features,train_acc,label='Training')
plt.plot(features,test_acc,label='Testing')
plt.xlabel('n° RFF')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.savefig('double_descent.png', dpi=150)
