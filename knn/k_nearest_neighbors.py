import numpy as np
import knn.vectorized_distance_metrics as dm


class KNNClassification:

    def __init__(self, k=1, metric="euclidean", tree=False):
        self.k = k
        self.tree = tree

        if metric == callable(metric):
            self.metric = metric
        else:
            metrics = {"euclidean": dm.euclidean,
                       "manhattan": dm.manhattan,
                       "hamming": dm.hamming,
                       "cosine": dm.cosine,
                       "pearson": dm.pearson,
                       "chisqr": dm.chisqr}
            # Default To Euclidean If Given Metric Does Not Exist
            self.metric = metrics.get(metric, dm.euclidean)

        self.labels = np.empty(0)
        self.train_data = np.empty(0)

    def train(self, labels, train_data):
        self.labels = labels
        self.train_data = train_data

        # Build Tree - Not Implemented Yet
        if self.tree:
            pass

    def predict(self, test_data):
        if self.tree:
            pass
        else:
            indices = self._brute_force_knn(test_data, False)
            labels = self.labels[indices]
            output_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, labels)
            return output_labels

    def _brute_force_knn(self, test_data, distances=False):
        distance_matrix = self.metric(self.train_data, test_data)
        print("Finished Distance Calc")
        k_smallest_ind = np.argpartition(distance_matrix, self.k-1)[:, :self.k]

        if distances:
            return k_smallest_ind, np.array([distance_matrix[i, x] for i, x in enumerate(k_smallest_ind)])

        return k_smallest_ind


# TODO - Implement KNN Regression
class KNNRegression:

    def __init__(self, k=1, metric="euclidean", tree=False):
        pass

    def train(self, train_response, train_data):
        pass

    def predict(self, test_data):
        pass



