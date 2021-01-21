
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn import metrics,pipeline,svm
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
n=100
train_X=np.random.rand(n)*0.5+0.1
train_X=np.expand_dims(train_X,axis=1)
train_y=np.sqrt(train_X*(1-train_X))*np.sin(1.1*np.pi/(train_X+0.05))


n=100
test_X=np.random.rand(n)*0.5+0.1
test_X=np.expand_dims(test_X,axis=1)
test_y=np.sqrt(test_X*(1-test_X))*np.sin(2.1*np.pi/(test_X+0.05))





for gamm in [500000]:
    feature_map_fourier = RBFSampler(gamma=gamm, random_state=1)
    fourier_approx_linear = pipeline.Pipeline([("feature_map", feature_map_fourier), ("svm",LinearRegression())])
    train_acc = []
    test_acc = []   
    features=np.linspace(1,300,301,dtype=int)
    for D in features:
        fourier_approx_linear.set_params(feature_map__n_components=D)
        fourier_approx_linear.fit(train_X, train_y)
        '''
        X_plot=np.expand_dims(np.linspace(0.1,0.6,10000),axis=1)
        y_plot=fourier_approx_linear.predict(X_plot)
        plt.plot(X_plot,y_plot,label=str(D))
        plt.scatter(train_X,train_y,label=str(D))
        plt.show()     '''
        train_acc.append(np.sqrt(np.mean((fourier_approx_linear.predict(train_X)-train_y)**2)))
        test_acc.append(np.sqrt(np.mean((fourier_approx_linear.predict(test_X)-test_y)**2)))
    plt.plot(features,train_acc,label='Training gamma ='+str(gamm))
    plt.plot(features,test_acc,label='Testing gamma ='+str(gamm))
plt.legend()
plt.xlabel('n° RFF')
plt.ylabel('L2 loss')
plt.grid()
plt.legend()
plt.savefig('double_descent_doppler.png', dpi=150)
plt.show()

