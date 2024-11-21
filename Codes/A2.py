"""
    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 4)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 4)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :Ytr type: numpy.ndarray of size (number_of_training_samples, 3)
    :Yts type: numpy.ndarray of size (number_of_test_samples, 3)
    :Ptrain_list type: List[numpy.ndarray]
    :Ptest_list type: List[numpy.ndarray]
    :w_list type: List[numpy.ndarray]
    :error_train_array type: numpy.ndarray
    :error_test_array type: numpy.ndarray
    """


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

def A2_A0200058W(N):
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=N)
    Ytr = OneHotEncoder(sparse=False).fit_transform(y_train.reshape(len(y_train), 1))
    Yts = OneHotEncoder(sparse=False).fit_transform(y_test.reshape(len(y_test), 1))

    Ptrain_list = list()
    Ptest_list = list()
    w_list = list()
    error_train_array = list()
    error_test_array = list()

    #training X_train, Ytr
    for i in range(1,11):
        P = PolynomialFeatures(i).fit_transform(X_train)
        Ptrain_list.append(P)

        if len(P) > len(P[0]): #m > d ie over determined
            LambdaIdentity = 0.0001 * np.eye(len(P[0]))
            w = np.linalg.inv(P.T @ P + LambdaIdentity) @ P.T @ Ytr #Primal
        else: # m < d ie under determined
            LambdaIdentity = 0.0001 * np.eye(len(P))
            w = P.T @ np.linalg.inv(P @ P.T + LambdaIdentity) @ Ytr #Duel

        w_list.append(w)

        prediction = [[1 if y == max(x) else 0 for y in x] for x in P @ w]
        check = Ytr - prediction #check if Ytr = prediction, 0 if same
        correct = np.where(~check.any(axis=1))[0]
        TrainError = len(check) - len(correct)
        error_train_array = np.append(error_train_array, np.array([TrainError]), axis=0)

    #testing X_test, Yts
    for i in range(1,11):
        P = PolynomialFeatures(i).fit_transform(X_test)
        Ptest_list.append(P)

        prediction = [[1 if y == max(x) else 0 for y in x] for x in P @ w_list[i - 1]]
        check = Yts - prediction #check if Ytr = prediction, 0 if same
        correct = np.where(~check.any(axis=1))[0]
        TestError = len(check) - len(correct)
        error_test_array = np.append(error_test_array, np.array([TestError]), axis=0)

    return X_train, X_test, y_train, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array


 A2_A0200058W(10)

'''
    print('X_train:\n{}'.format(X_train))
    print('X_test:\n{}'.format(X_test))
    print('y_train:\n{}'.format(y_train))
    print('y_test:\n{}'.format(y_test))
    print('Ytr:\n{}'.format(Ytr))
    print('Yts:\n{}'.format(Yts))
    print('Ptrain_list:\n{}'.format(Ptrain_list))
    print('Ptest_list:\n{}'.format(Ptest_list))
    print('w_list:\n{}'.format(w_list))
    print('error_train_array:\n{}'.format(error_train_array))
    print('error_test_array:\n{}'.format(error_test_array))
    print(type(X_train))
    print(type(X_test))
    print(type(y_train))
    print(type(y_test))
    print(type(Ytr))
    print(type(Yts))
    print(type(Ptrain_list))
    print(type(Ptest_list))
    print(type(w_list))
    print(type(error_train_array))
    print(type(error_test_array))
'''