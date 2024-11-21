'''
  Input type
  :N type: int
  :TestSize type: float
  :MaxTreeDepth type: int

Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 8)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 8)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :ytr_Tree_list type: List[numpy.ndarray]
    :yts_Tree_list type: List[numpy.ndarray]
    :mse_trainTree_array type: numpy.ndarray
    :mse_testTree_array type: numpy.ndarray
    :ytr_Forest_list type: List[numpy.ndarray]
    :yts_Forest_list type: List[numpy.ndarray]
    :mse_trainForest_array type: numpy.ndarray
    :mse_testForest_array type: numpy.ndarray
'''


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def A3_A0200058W(N, TestSize, MaxTreeDepth):
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TestSize, random_state=N)

    ytr_Tree_list = list()
    yts_Tree_list = list()
    mse_trainTree_array = list()
    mse_testTree_array = list()
    for i in range(1, MaxTreeDepth + 1):
        tree_regressor = DecisionTreeRegressor(criterion='mse', max_depth=i, random_state=0)
        tree_regressor.fit(X_train, y_train)

        ytr_Tree_list.append(tree_regressor.predict(X_train))
        yts_Tree_list.append(tree_regressor.predict(X_test))
        mse_trainTree_array.append(mean_squared_error(y_train, tree_regressor.predict(X_train)))
        mse_testTree_array.append(mean_squared_error(y_test, tree_regressor.predict(X_test)))

    mse_trainTree_array = np.array(mse_trainTree_array)
    mse_testTree_array = np.array(mse_testTree_array)

    ytr_Forest_list = list()
    yts_Forest_list = list()
    mse_trainForest_array = list()
    mse_testForest_array = list()
    for j in range(1, MaxTreeDepth + 1):
        forest_regressor = RandomForestRegressor(criterion='mse', max_depth=j, random_state=0)
        forest_regressor.fit(X_train, y_train)

        ytr_Forest_list.append(forest_regressor.predict(X_train))
        yts_Forest_list.append(forest_regressor.predict(X_test))
        mse_trainForest_array.append(mean_squared_error(y_train, forest_regressor.predict(X_train)))
        mse_testForest_array.append(mean_squared_error(y_test, forest_regressor.predict(X_test)))

    mse_trainForest_array = np.array(mse_trainForest_array)
    mse_testForest_array = np.array(mse_testForest_array)

    return X_train, y_train, X_test, y_test, ytr_Tree_list, yts_Tree_list, mse_trainTree_array, mse_testTree_array, ytr_Forest_list, yts_Forest_list, mse_trainForest_array, mse_testForest_array

'''
    print('X_train:\n{}'.format(X_train))
    print('X_test:\n{}'.format(X_test))
    print('y_train:\n{}'.format(y_train))
    print('y_test:\n{}'.format(y_test))
    
    print('ytr_Tree_list:\n{}'.format(ytr_Tree_list))
    print('yts_Tree_list:\n{}'.format(yts_Tree_list))
    print('mse_trainTree_array:\n{}'.format(mse_trainTree_array))
    print('mse_testTree_array:\n{}'.format(mse_testTree_array))
    
    print('ytr_Forest_list:\n{}'.format(ytr_Forest_list))
    print('yts_Forest_list:\n{}'.format(yts_Forest_list))
    print('mse_trainForest_array:\n{}'.format(mse_trainForest_array))
    print('mse_testForest_array:\n{}'.format(mse_testForest_array))
'''