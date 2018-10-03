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
test_labels = test_data[:200, 0]
test_data = test_data[:200, 1:]
train_labels = train_data[:6000, 0]
train_data = train_data[:6000, 1:]

# Create and Train Classifier
knn = KNNClassification(k=5, metric="euclidean")
knn.train(train_labels, train_data)

# Get Predictions
predictions = knn.predict(test_data)

print("Test Labels:\n" + str(test_labels))
print("Predicted Labels:\n" + str(predictions))

accuracy = sum(test_labels == predictions)/test_labels.size
print("Accuracy: " + str(accuracy))
