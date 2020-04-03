from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier  # importul clasei
from sklearn import preprocessing

import numpy as np


# ------ exercitiul 1
def plot3d_data(X, y):
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], 'b');
    ax.scatter3D(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], 'r');
    plt.show()


def plot3d_data_and_decision_function(X, y, W, b):
    ax = plt.axes(projection='3d')
    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))
    # calculate corresponding z
    # [x, y, z] * [coef1, coef2, coef3] + b = 0
    zz = (-W[0] * xx - W[1] * yy - b) / W[2]
    ax.plot_surface(xx, yy, zz, alpha=0.5)
    ax.scatter3D(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], 'b');
    ax.scatter3D(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], 'r');
    plt.show()


# incarcarea datelor de antrenare
X = np.loadtxt('./data/3d-points/x_train.txt')
y = np.loadtxt('./data/3d-points/y_train.txt', 'int')

plot3d_data(X, y)
# incarcarea datelor de testare
X_test = np.loadtxt('./data/3d-points/x_test.txt')
y_test = np.loadtxt('./data/3d-points/y_test.txt', 'int')

from sklearn.neural_network import MLPClassifier  # importul clasei

mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(100,),
                                     activation='relu', solver='sgd', alpha=0.0001, batch_size='auto',
                                     learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
                                     max_iter=200, shuffle=True, random_state=None, tol=0.0001,
                                     momentum=0.9, early_stopping=False, validation_fraction=0.1,
                                     n_iter_no_change=10)

from sklearn.linear_model import Perceptron  # importul clasei

perceptron_model = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True,
                              max_iter=None, tol=None, shuffle=True, eta0=1.0)

Sc = preprocessing.StandardScaler()
Sc.fit(X)
x_train_sc = Sc.transform(X)

Sc.fit(X_test)
x_test_sc = Sc.transform(X_test)

perc = Perceptron(eta0=0.1, tol=1e-5)
perc.fit(x_train_sc, y)
print(perc.score(x_test_sc, y_test))
print(perc.coef_)
print (perc.intercept_)
plot3d_data_and_decision_function(X, y, perc.coef_[0], perc.intercept_)

