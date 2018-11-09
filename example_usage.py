import timeit

import numpy as np
from knn.k_nearest_neighbors import KNNClassification, KNNRegression


# '''
# Simple Example - Classification
# '''
#
# # Sample Data
# train_data = np.array([[1., 2., 3.],
#                        [5., 6., 7.],
#                        [8., 9., 10.]])
# train_labels = np.array([0, 1, 1])
#
# test_data = np.array([[5., 10., 15.],
#                       [10., 20., 30.]])
#
# # Create and Train Classifier
# knn = KNNClassification(k=1, metric="euclidean")
# knn.train(train_labels, train_data)
#
# # Get Predictions
# predictions = knn.predict(test_data)
# print("k=1 Predictions:\n" + str(predictions))
#
#
# '''
# MNIST Classification
# '''
# # Load Data
# mnist_data = np.load('./sample_data/mnist/mnist_data.npz')
# train_data = mnist_data['train_data']
# test_data = mnist_data['test_data']
#
# # Subset Data If Desired
# test_labels = test_data[:20000, 0]
# test_data = test_data[:20000, 1:].astype(np.float)
# train_labels = train_data[:60000, 0]
# train_data = train_data[:60000, 1:].astype(np.float)
#
# # Create and Train Classifier
# knn = KNNClassification(k=5, metric="manhattan")
# knn.train(train_labels, train_data)
#
# # Get Predictions
# predictions = knn.predict(test_data)
#
# print("Test Labels:\n" + str(test_labels))
# print("Predicted Labels:\n" + str(predictions))
#
# accuracy = sum(test_labels == predictions)/test_labels.size
# print("Accuracy: " + str(accuracy))
#
#
# '''
# Using A Distance Metric From Another Package
# '''
# from scipy.spatial import distance
#
# train_data = np.array([[1., 2., 3.],
#                        [5., 6., 7.],
#                        [8., 9., 10.]])
# train_labels = np.array([0, 1, 1])
#
# test_data = np.array([[5., 10., 15.],
#                       [10., 20., 30.]])
#
# # Needed To Swap Input Order And Set Metric Argument
# def scipy_cityblock(vectors_a, vectors_b):
#     return distance.cdist(vectors_b, vectors_a, 'cityblock')
#
# # Create and Train Classifier
# knn = KNNClassification(k=5, metric=scipy_cityblock)
# knn.train(train_labels, train_data)
#
# # Get Predictions
# predictions = knn.predict(test_data)
#
# print("Test Labels:\n" + str(test_labels))
# print("Predicted Labels:\n" + str(predictions))
#
# accuracy = sum(test_labels == predictions)/test_labels.size
# print("Accuracy: " + str(accuracy))
#
#
# '''
# Million Song Dataset(MSD) Regression
# '''
# msd_data = np.load('./sample_data/msd/msd_data.npz')
#
# train_data = msd_data['train_data']
# train_response = train_data[:600000, 0]
# train_data = train_data[:600000, 1:]
#
# test_data = msd_data['test_data']
# test_response = test_data[:500, 0]
# test_data = test_data[:500, 1:]
#
# knn = KNNRegression(k=35, metric="manhattan")
# knn.train(train_response, train_data)
#
# # Get Predictions
# predicted_response = knn.predict(test_data)
#
# print("Actual Response:\n" + str(test_response[:20]))
# print("Predicted Response:\n" + str(predicted_response[:20]))
#
# error = np.sqrt(np.mean(np.square(test_response-predicted_response)))
# print("Root Mean Square Error: " + str(error))
#
#
# '''
#  Simple Example - Classification Using Ball Tree
# '''
#
# train_data = np.array([[4, -2], [5, 5], [8, 7], [-6, -1], [-1, -3], [-4,-8]])
# train_labels = np.array([1, 1, 1, 0, 0, 0])
#
# test_data = np.array([[6, 4], [-8, -4]])
#
# knn = KNNClassification(k=3, metric="euclidean", tree=True, tree_leaf_size=3)
# knn.train(train_labels, train_data)
# predictions = knn.predict(test_data)
# print("k=1 Predictions:\n" + str(predictions))


'''
MNIST Classification - Ball Tree
'''
# # Load Data
# mnist_data = np.load('./sample_data/mnist/mnist_data.npz')
# train_data = mnist_data['train_data']
# test_data = mnist_data['test_data']
#
# # Subset Data If Desired
# test_labels = test_data[:1000, 0]
# test_data = test_data[:1000, 1:].astype(np.float)
# train_labels = train_data[:1000, 0]
# train_data = train_data[:1000, 1:].astype(np.float)
#
# # Create and Train Classifier
# knn = KNNClassification(k=3, metric="manhattan", tree=True, tree_leaf_size=100)
# knn.train(train_labels, train_data)
#
# # Get Predictions
# predictions = knn.predict(test_data)
#
# print("Test Labels:\n" + str(test_labels))
# print("Predicted Labels:\n" + str(predictions))
#
# accuracy = sum(test_labels == predictions)/test_labels.size
# print("Accuracy: " + str(accuracy))


# '''
# Million Song Dataset(MSD) Regression
# '''
# msd_data = np.load('./sample_data/msd/msd_data.npz')
#
# train_data = msd_data['train_data']
# train_response = train_data[:600, 0]
# train_data = train_data[:600, 1:]
#
# test_data = msd_data['test_data']
# test_response = test_data[:500, 0]
# test_data = test_data[:500, 1:]
#
# knn = KNNRegression(k=35, metric="manhattan", tree=True, tree_leaf_size=3)
# knn.train(train_response, train_data)
#
# # Get Predictions
# predicted_response = knn.predict(test_data)
#
# print("Actual Response:\n" + str(test_response[:20]))
# print("Predicted Response:\n" + str(predicted_response[:20]))
#
# error = np.sqrt(np.mean(np.square(test_response-predicted_response)))
# print("Root Mean Square Error: " + str(error))



from knn.ball_tree import BallTree

mnist_data = np.load('./sample_data/mnist/mnist_data.npz')
train_data = mnist_data['train_data']
test_data = mnist_data['test_data']

# Subset Data If Desired
test_labels = test_data[:, 0]
test_data = test_data[:1000, 1:4].astype(np.float)
train_labels = train_data[:, 0]
train_data = train_data[:1000, 1:4].astype(np.float)

tree = BallTree(train_data, 100)
tree.build_tree()


tree.query(test_data, 3)




# print("Test Labels:\n" + str(test_labels))
# print("Predicted Labels:\n" + str(train_labels[tree.heap_inds.astype(np.int)]))


#accuracy = sum(test_labels == predictions)/test_labels.size
#print("Accuracy: " + str(accuracy))