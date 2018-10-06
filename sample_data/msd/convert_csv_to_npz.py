import numpy as np

# Read In MSD Data From TXT Files
# Data From : https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
msd_data = np.genfromtxt('./YearPredictionMSD.txt', delimiter=',')


# Subset Into Training and Testing Data
train_data = msd_data[0:463715, :]
test_data = msd_data[463715:, :]

# Convert To npz
np.savez('./msd_data.npz', train_data=train_data, test_data=test_data)
