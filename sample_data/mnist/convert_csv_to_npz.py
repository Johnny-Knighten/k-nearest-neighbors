import numpy as np

# Read In MNIST Data From CSV Files
# Data From : https://pjreddie.com/projects/mnist-in-csv/
test_data = np.genfromtxt('./mnist_test.csv', delimiter=',')
train_data = np.genfromtxt('./mnist_train.csv', delimiter=',')

# Convert To int32 To Save Some Room
test_data = test_data.astype(np.int32)
train_data = train_data.astype(np.int32)

# Convert To npz
np.savez('./mnist_data.npz', train_data=train_data, test_data=test_data)
