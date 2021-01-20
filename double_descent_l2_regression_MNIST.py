
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
train_X=train_X[0:3000,:]
train_y=train_y[0:3000]
train_y_onehot = np.zeros((train_y.size, train_y.max()+1))
train_y_onehot[np.arange(train_y.size),train_y] = 1

test_X=test_X.reshape(-1,28*28)/(28*28)
test_X=test_X[0:3000,:]
test_y=test_y[0:3000]
test_y_onehot = np.zeros((test_y.size, test_y.max()+1))
test_y_onehot[np.arange(test_y.size),test_y] = 1

train_acc = []
test_acc = []

feature_map_fourier = RBFSampler(gamma=.2, random_state=1)
fourier_approx_linear = pipeline.Pipeline([("feature_map", feature_map_fourier), ("svm",LinearRegression())])

features=[1000,3000]#,2500,2750,2850,2900,2950,2975,3000,3025,3050,3100,3150,3250,3500,4000]
for D in features:
    fourier_approx_linear.set_params(feature_map__n_components=D)
    fourier_approx_linear.fit(train_X, train_y_onehot)
    train_acc.append(fourier_approx_linear.score(train_X, train_y_onehot))
    test_acc.append(fourier_approx_linear.score(test_X, test_y_onehot))
plt.xlabel('n° RFF')
plt.ylabel('Accuracy')
plt.yscale('log')
plt.plot(features,train_acc,label='Training')
plt.plot(features,test_acc,label='Testing')
plt.grid()
plt.legend()
plt.savefig('double_descent_linear.png', dpi=150)
