import unittest
import numpy as np
import data_storage as ds
import decision_trees as dt

class TestMethods(unittest.TestCase):
    # tests building numpy array
    def test_build_nparray(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        testArray = [0., 0., 0., 0., 0., 0.]
        self.assertNotEqual(ds.build_nparray(testData), testArray)
        
    # tests building list
    def test_build_list(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        self.assertTrue(ds.build_nparray(testData))
        
    # tests building dict
    def test_build_dict(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        self.assertTrue(ds.build_dict(testData))

    # tests training data
    def test_train_data(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        self.assertTrue(dt.DT_train_binary(X,Y,max_depth))

    # tests tree result
    def test_tree(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.DT_test_binary(X,Y,DT))

    # tests making prediction
    def test_make_prediction(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.DT_make_prediction(Y,DT))

    # tests building tree
    def test_build_tree(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.build_tree(X,Y,max_depth,DT))

    # tests building tree with data for different edge cases
    def test_build_tree_data2(self):
        file_name = "data2.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.build_tree(X,Y,max_depth,DT))

    # tests building tree with data for different edge cases
    def test_build_tree_data3(self):
        file_name = "data3.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.build_tree(X,Y,max_depth,DT))

    # tests building tree with data for different edge cases
    def test_build_tree_data4(self):
        file_name = "data4.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        self.assertTrue(dt.build_tree(X,Y,max_depth,DT))

    def test_integration_build_nparray(self):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        expected_data_array = np.array([[5., 6., 7.], [9., 10., 11.]])
        expected_label_array = np.array([8, 12])
        data_array, label_array = ds.build_nparray(data)
        self.assertEqual(data_array[0][0], expected_data_array[0][0])
        self.assertEqual(label_array[0], expected_label_array[0])

    def test_integration_build_list(self):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        expected_data_list = [[5.0, 6.0, 7.0], [9.0, 10.0, 11.0]]
        expected_label_list = [8, 12]
        data_list, label_list = ds.build_list(data)
        self.assertEqual(data_list, expected_data_list)
        self.assertEqual(label_list, expected_label_list)

    def test_integration_build_dict(self):
        data = [['a', 'b', 'c', 'd', 'label'],
                ['1', '2', '3', '4', '0'],
                ['5', '6', '7', '8', '1'],
                ['9', '10', '11', '12', '0']]
        expected_samples = {
            0: {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0},
            1: {'a': 5.0, 'b': 6.0, 'c': 7.0, 'd': 8.0},
            2: {'a': 9.0, 'b': 10.0, 'c': 11.0, 'd': 12.0}
        }
        expected_label_dict = {0: 0, 1: 1, 2: 0}
        samples, label_dict = ds.build_dict(data)
        self.assertEqual(samples, expected_samples)
        self.assertEqual(label_dict, expected_label_dict)
        
    def test_integration_tree_prediciton(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        predicition = dt.DT_make_prediction(X[1], DT)
        self.assertEqual(predicition, Y[0])

    def test_integration_accuracy(self):
        file_name = "data1.csv"
        testData = np.genfromtxt(file_name, dtype=str, delimiter=',')
        max_depth = 3
        X,Y = ds.build_nparray(testData)
        DT = dt.DT_train_binary(X,Y,max_depth)
        acc = dt.DT_test_binary(X,Y,DT)
        assert acc >= 0 and acc <= 1

