import pytest
import numpy as np
import data_storage as ds
import decision_trees as dt

# tests building numpy array
def test_build_nparray():
    file_name = "data1.csv"
    testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
    assertTrue(ds.build_nparray(testData))

# tests building list
def test_build_list():
    file_name = "data1.csv"
    testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
    assertTrue(ds.build_list(testData))

# tests building dict
def test_build_dict():
    file_name = "data1.csv"
    testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
    assertTrue(ds.build_dict(testData))

# tests training data
def test_train_data():
    file_name = "data1.csv"
    testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
    max_depth = 3
    X,Y = ds.build_nparray(testData)
    assertTrue(dt.DT_train_binary(X,Y,max_depth))

# tests tree result
def test_tree():
    file_name = "data1.csv"
    testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
    max_depth = 3
    X,Y = ds.build_nparray(testData)
    DT = dt.DT_train_binary(X,Y,max_depth)
    assertTrue(dt.DT_test_binary(X,Y,DT))

# tests making prediction
def test_make_prediction():
    file_name = "data1.csv"
    testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
    max_depth = 3
    X,Y = ds.build_nparray(testData)
    DT = dt.DT_train_binary(X,Y,max_depth)
    assertTrue(dt.DT_make_prediction(Y,DT))

# tests building tree
def test_build_tree():
    file_name = "data1.csv"
    testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
    max_depth = 3
    X,Y = ds.build_nparray(testData)
    DT = dt.DT_train_binary(X,Y,max_depth)
    assertTrue(dt.build_tree(X,Y,max_depth,DT))

# tests building tree with data for different edge cases
def test_build_tree_data2():
    file_name = "data2.csv"
    testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
    max_depth = 3
    X,Y = ds.build_nparray(testData)
    DT = dt.DT_train_binary(X,Y,max_depth)
    assertTrue(dt.build_tree(X,Y,max_depth,DT))

# tests building tree with data for different edge cases
def test_build_tree_data3():
    file_name = "data3.csv"
    testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
    max_depth = 3
    X,Y = ds.build_nparray(testData)
    DT = dt.DT_train_binary(X,Y,max_depth)
    assertTrue(dt.build_tree(X,Y,max_depth,DT))

# tests building tree with data for different edge cases
def test_build_tree_data4():
    file_name = "data4.csv"
    testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
    max_depth = 3
    X,Y = ds.build_nparray(testData)
    DT = dt.DT_train_binary(X,Y,max_depth)
    assertTrue(dt.build_tree(X,Y,max_depth,DT))
