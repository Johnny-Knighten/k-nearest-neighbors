import numpy as np
from knn.k_nearest_neighbors import KNNClassification, KNNRegression



'''
Simple Example - Classification
'''

# Sample Data
train_data = np.array([[1., 2., 3.],
                       [5., 6., 7.],
                       [8., 9., 10.]])
train_labels = np.array([0, 1, 1])

test_data = np.array([[5., 10., 15.],
                      [10., 20., 30.]])

# Create and Train Classifier
knn = KNNClassification(use_tree=True, metric="euclidean")
knn.train(train_labels, train_data)

# Get Predictions
predictions = knn.predict(test_data, k=1)
print("k=1 Predictions:\n" + str(predictions))


# '''
# MNIST Classification
# '''
# # Load Data
# mnist_data = np.load('./sample_data/mnist/mnist_data.npz')
# train_data = mnist_data['train_data']
# test_data = mnist_data['test_data']
#
# # Subset Data If Desired
# test_labels = test_data[:2000, 0]
# test_data = test_data[:2000, 1:].astype(np.float)
# train_labels = train_data[:6000, 0]
# train_data = train_data[:6000, 1:].astype(np.float)
#
# # Create and Train Classifier
# knn = KNNClassification(metric="manhattan")
# knn.train(train_labels, train_data)
#
# # Get Predictions
# predictions = knn.predict(test_data, k=5)
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
# knn = KNNClassification(metric=scipy_cityblock)
# knn.train(train_labels, train_data)
#
# # Get Predictions
# predictions = knn.predict(test_data, k=1)
#
# print("k=1 Predictions:\n" + str(predictions))
#
#
# '''
# Million Song Dataset(MSD) Regression
# '''
# msd_data = np.load('./sample_data/msd/msd_data.npz')
#
# # Note - astype() Used To Make Arrays Contiguous
# train_data = msd_data['train_data']
# train_response = train_data[:600000, 0]
# train_data = train_data[:600000, 1:].astype(np.float)
#
# test_data = msd_data['test_data']
# test_response = test_data[:500, 0]
# test_data = test_data[:500, 1:].astype(np.float)
#
# knn = KNNRegression(metric="manhattan")
# knn.train(train_response, train_data)
#
# # Get Predictions
# predicted_response = knn.predict(test_data, k=35)
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
# train_data = np.array([[4, -2], [5, 5], [8, 7], [-6, -1], [-1, -3], [-4,-8]]).astype(np.float)
# train_labels = np.array([1, 1, 1, 0, 0, 0])
#
# test_data = np.array([[6, 4], [-8, -4]]).astype(np.float)
#
# knn = KNNClassification(metric="euclidean", use_tree=True, tree_leaf_size=3)
# knn.train(train_labels, train_data)
# predictions = knn.predict(test_data, k=3)
# print("k=3 Predictions:\n" + str(predictions))
#
#
# '''
# MNIST Classification - Ball Tree
# '''
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
# knn = KNNClassification(metric="manhattan", use_tree=True, tree_leaf_size=100)
# knn.train(train_labels, train_data)
#
# # Get Predictions
# predictions = knn.predict(test_data, k=3)
#
# print("Test Labels:\n" + str(test_labels))
# print("Predicted Labels:\n" + str(predictions))
#
# accuracy = sum(test_labels == predictions)/test_labels.size
# print("Accuracy: " + str(accuracy))
#
#
# '''
# Million Song Dataset(MSD) Regression - Ball Tree
# '''
# msd_data = np.load('./sample_data/msd/msd_data.npz')
#
# train_data = msd_data['train_data']
# train_response = train_data[:60000, 0]
# train_data = train_data[:60000, 1:].astype(np.float)
#
# test_data = msd_data['test_data']
# test_response = test_data[:500, 0]
# test_data = test_data[:500, 1:].astype(np.float)
#
# knn = KNNRegression(metric="manhattan", use_tree=True, tree_leaf_size=3)
# knn.train(train_response, train_data)
#
# # Get Predictions
# predicted_response = knn.predict(test_data, k=35)
#
# print("Actual Response:\n" + str(test_response[:20]))
# print("Predicted Response:\n" + str(predicted_response[:20]))
#
# error = np.sqrt(np.mean(np.square(test_response-predicted_response)))
# print("Root Mean Square Error: " + str(error))
#
#
# '''
#  Structured Data
# '''
# # Ball Trees Benefit Heavily If Data Is Has Structure (Naturally Partitioned Into Hyper Spheres "Balls")
#
# points_per_region = 200
# number_of_regions = 16
#
# random_data = np.empty((points_per_region * number_of_regions,2))
# region = 0
#
# for i in range(-30, 30, 15):
#     for j in range(-30, 30, 15):
#         mu = np.array([i+7.5, j+7.5])
#         sigma = np.array([[1, 0], [0, 1]])
#         random_data[region:region+points_per_region] = np.random.multivariate_normal(mu, sigma, points_per_region)
#         region += points_per_region
#
# labels = np.repeat(np.arange(number_of_regions), points_per_region)
#
# knn = KNNClassification(use_tree=True, tree_leaf_size=20, metric="euclidean")
# knn.train(labels, random_data)
# # Get Predictions
# predictions = knn.predict(random_data, k=5)
