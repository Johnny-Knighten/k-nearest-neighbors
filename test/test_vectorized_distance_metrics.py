import unittest
import numpy as np
from math import sqrt

from knn.vectorized_distance_metrics import euclidean, manhattan, cosine, pearson, hamming, chisqr


class TestVectorizedDistanceMetrics(unittest.TestCase):

    def setUp(self):
        # Vectors That Range From 0 Degrees to 360 Degrees By 90 Degree Increments
        self.origin_2_dim = np.array([0.0, 0.0])
        self.zero_degree_2_dim = np.array([1.0, 0.0])
        self.ninety_degree_2_dim = np.array([0.0, 1.0])
        self.one_eighty_degree_2_dim = np.array([-1.0, 0.0])
        self.two_seventy_degree_2_dim = np.array([0.0, -1.0])

        # Vectors That Range From 45 Degrees to 315 Degrees By 90 Degree Increments
        self.forty_five_degree_2_dim = np.array([sqrt(2)/2.0, sqrt(2)/2.0])
        self.one_thirty_five_degree_2_dim = np.array([-sqrt(2)/2.0, sqrt(2)/2.0])
        self.two_twenty_five_degree_2_dim = np.array([-sqrt(2)/2.0, -sqrt(2)/2.0])
        self.three_fifteen_degree_2_dim = np.array([sqrt(2)/2.0, -sqrt(2)/2.0])

        self.test_train_vectors = np.array([self.origin_2_dim, self.zero_degree_2_dim, self.ninety_degree_2_dim,
                                            self.one_eighty_degree_2_dim, self.two_seventy_degree_2_dim])
        self.test_test_vectors = np.array([self.forty_five_degree_2_dim, self.one_thirty_five_degree_2_dim,
                                           self.two_twenty_five_degree_2_dim, self.three_fifteen_degree_2_dim])

        # Bit Strings For Hamming Distance
        self.bs_all_1_len_6 = np.array([1, 1, 1, 1, 1, 1])
        self.bs_all_0_len_6 = np.array([0, 0, 0, 0, 0, 0])
        self.bs_1_on_5_off = np.array([1, 0, 0, 0, 0, 0])
        self.bs_2_on_4_off = np.array([1, 1, 0, 0, 0, 0])
        self.bs_3_on_3_off = np.array([1, 1, 1, 0, 0, 0])
        self.bs_4_on_2_off = np.array([1, 1, 1, 1, 0, 0])
        self.bs_5_on_1_off = np.array([1, 1, 1, 1, 1, 0])

        self.test_bs_vectors = np.array([self.bs_all_1_len_6, self.bs_all_0_len_6])
        self.train_bs_vectors = np.array([self.bs_1_on_5_off, self.bs_2_on_4_off, self.bs_3_on_3_off,
                                          self.bs_4_on_2_off, self.bs_5_on_1_off])

        # All Int Vectors For Chi-Squared
        self.chi_vector_1 = np.array([5., 10., 3., 7., 7.])
        self.chi_vector_2 = np.array([10., 5., 8., 2., 5.])
        self.chi_vector_3 = np.array([10., 10., 4., 4., 1.])
        self.chi_vector_4 = np.array([2., 1., 9., 5., 3.])
        self.chi_vector_5 = np.array([7., 8., 5., 1., 9.])

        self.test_chi_vectors = np.array([self.chi_vector_1, self.chi_vector_2])
        self.train_chi_vectors = np.array([self.chi_vector_3, self.chi_vector_4, self.chi_vector_5])

    def test_euclidean(self):
        distance_matrix = euclidean(self.test_train_vectors, self.test_test_vectors)
        self.assertTrue(np.allclose(np.array([1., 0.76536686, 0.76536686, 1.84775907, 1.84775907]),
                                    distance_matrix[0, :]))
        self.assertTrue(np.allclose(np.array([1., 1.84775907, 0.76536686, 0.76536686, 1.84775907]),
                                    distance_matrix[1, :]))
        self.assertTrue(np.allclose(np.array([1., 1.84775907, 1.84775907, 0.76536686, 0.76536686]),
                                    distance_matrix[2, :]))
        self.assertTrue(np.allclose(np.array([1., 0.76536686, 1.84775907, 1.84775907, 0.76536686]),
                                    distance_matrix[3, :]))

    def test_manhattan(self):
        distance_matrix = manhattan(self.test_train_vectors, self.test_test_vectors)
        self.assertTrue(np.allclose(np.array([1.41421356, 1., 1., 2.41421356, 2.41421356]), distance_matrix[0, :]))
        self.assertTrue(np.allclose(np.array([1.41421356, 2.41421356, 1., 1., 2.41421356]), distance_matrix[1, :]))
        self.assertTrue(np.allclose(np.array([1.41421356, 2.41421356, 2.41421356, 1., 1.]), distance_matrix[2, :]))
        self.assertTrue(np.allclose(np.array([1.41421356, 1., 2.41421356, 2.41421356, 1.]), distance_matrix[3, :]))

    def test_hamming(self):
        distance_matrix = hamming(self.train_bs_vectors, self.test_bs_vectors)
        self.assertTrue(np.allclose(np.array([5, 4, 3, 2, 1]), distance_matrix[0, :]))
        self.assertTrue(np.allclose(np.array([1, 2, 3, 4, 5]), distance_matrix[1, :]))

    def test_cosine(self):
        distance_matrix = cosine(self.test_train_vectors, self.test_test_vectors)
        self.assertEqual(np.sum(np.isnan(distance_matrix[:, 0])), 4)
        self.assertTrue(np.allclose(np.array([0.29289322, 0.29289322, 1.70710678, 1.70710678]), distance_matrix[0, 1:]))
        self.assertTrue(np.allclose(np.array([1.70710678, 0.29289322, 0.29289322, 1.70710678]), distance_matrix[1, 1:]))
        self.assertTrue(np.allclose(np.array([1.70710678, 1.70710678, 0.29289322, 0.29289322]), distance_matrix[2, 1:]))
        self.assertTrue(np.allclose(np.array([0.29289322, 1.70710678, 1.70710678, 0.29289322]), distance_matrix[3, 1:]))

    def test_pearson(self):
        distance_matrix = pearson(self.test_train_vectors, self.test_test_vectors)
        self.assertTrue(np.allclose(np.array([1., 1., 1., 1., 1.]), distance_matrix[0, :]))
        self.assertTrue(np.allclose(np.array([1., 2., 0., 0., 2.]), distance_matrix[1, :]))
        self.assertTrue(np.allclose(np.array([1., 1., 1., 1., 1.]), distance_matrix[2, :]))
        self.assertTrue(np.allclose(np.array([1., 0., 2., 2., 0.]), distance_matrix[3, :]))

    def test_chisqr(self):
        distance_matrix = chisqr(self.train_chi_vectors, self.test_chi_vectors)
        self.assertTrue(np.allclose(np.array([0.08683298, 0.1335905, 0.07737234]), distance_matrix[0, :]))
        self.assertTrue(np.allclose(np.array([0.08521912, 0.11670043, 0.06137515]), distance_matrix[1, :]))

if __name__ == '__main__':
    unittest.main()
