import numpy as np
from knn.k_nearest_neighbors import KNNClassification


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
knn = KNNClassification(k=1, metric="euclidean")
knn.train(train_labels, train_data)

# Get Predictions
predictions = knn.predict(test_data)
print("k=1 Predictions:\n" + str(predictions))


'''
MNIST Classification
'''
# Load Data
mnist_data = np.load('./sample_data/mnist/mnist_data.npz')
train_data = mnist_data['train_data']
test_data = mnist_data['test_data']

# Subset Data If Desired
test_labels = test_data[:20000, 0]
test_data = test_data[:20000, 1:].astype(np.float)
train_labels = train_data[:60000, 0]
train_data = train_data[:60000, 1:].astype(np.float)

print(type(test_data.shape[0]))

# Create and Train Classifier
knn = KNNClassification(k=5, metric="manhattan")
knn.train(train_labels, train_data)

# Get Predictions
predictions = knn.predict(test_data)

print("Test Labels:\n" + str(test_labels))
print("Predicted Labels:\n" + str(predictions))

accuracy = sum(test_labels == predictions)/test_labels.size
print("Accuracy: " + str(accuracy))
#
# '''
# Using A Distance Metric From Another Package
# '''
from scipy.spatial import distance

train_data = np.array([[1., 2., 3.],
                       [5., 6., 7.],
                       [8., 9., 10.]])
train_labels = np.array([0, 1, 1])

test_data = np.array([[5., 10., 15.],
                      [10., 20., 30.]])


# Needed To Swap Input Order And Set Metric Argument
def scipy_cityblock(vectors_a, vectors_b):
    return distance.cdist(vectors_b, vectors_a, 'cityblock')

# Create and Train Classifier
knn = KNNClassification(k=5, metric=scipy_cityblock)
knn.train(train_labels, train_data)

# Get Predictions
predictions = knn.predict(test_data)

print("Test Labels:\n" + str(test_labels))
print("Predicted Labels:\n" + str(predictions))

accuracy = sum(test_labels == predictions)/test_labels.size
print("Accuracy: " + str(accuracy))
