import unittest
import numpy as np
from math import sqrt, cos, pi
from knn.distance_metrics import euclidean, manhattan, hamming, chisqr, cosine, pearson


class TestDistanceMetrics(unittest.TestCase):

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

        # For Chi-Squared
        self.chi_vector_1 = np.array([5.0, 10.0])
        self.chi_vector_2 = np.array([10.0, 5.0])

        # Bit Strings
        self.bs_all_1_len_6 = np.array([1, 1, 1, 1, 1, 1]).astype(np.float)
        self.bs_all_0_len_6 = np.array([0, 0, 0, 0, 0, 0]).astype(np.float)
        self.bs_1_on_5_off = np.array([1, 0, 0, 0, 0, 0]).astype(np.float)
        self.bs_2_on_4_off = np.array([1, 1, 0, 0, 0, 0]).astype(np.float)
        self.bs_3_on_3_off = np.array([1, 1, 1, 0, 0, 0]).astype(np.float)
        self.bs_4_on_2_off = np.array([1, 1, 1, 1, 0, 0]).astype(np.float)
        self.bs_5_on_1_off = np.array([1, 1, 1, 1, 1, 0]).astype(np.float)

    def test_euclidean(self):
        # Identical Input
        self.assertEqual(euclidean(self.origin_2_dim, self.origin_2_dim), 0)

        # 90 Degree Rotations Around Origin Starting At 0 Degrees
        self.assertEqual(euclidean(self.origin_2_dim, self.zero_degree_2_dim), 1)
        self.assertEqual(euclidean(self.origin_2_dim, self.ninety_degree_2_dim), 1)
        self.assertEqual(euclidean(self.origin_2_dim, self.one_eighty_degree_2_dim), 1)
        self.assertEqual(euclidean(self.origin_2_dim, self.two_seventy_degree_2_dim), 1)

        # 90 Degree Rotations Around Origin Starting At 45 Degrees
        self.assertEqual(euclidean(self.origin_2_dim, self.forty_five_degree_2_dim), 1)
        self.assertEqual(euclidean(self.origin_2_dim, self.one_thirty_five_degree_2_dim), 1)
        self.assertEqual(euclidean(self.origin_2_dim, self.two_twenty_five_degree_2_dim), 1)
        self.assertEqual(euclidean(self.origin_2_dim, self.three_fifteen_degree_2_dim), 1)

    def test_manhattan(self):
        # Identical Input
        self.assertEqual(manhattan(self.origin_2_dim, self.origin_2_dim), 0)

        # 90 Degree Rotations Around Origin Starting At 0 Degrees
        self.assertEqual(manhattan(self.origin_2_dim, self.zero_degree_2_dim), 1)
        self.assertEqual(manhattan(self.origin_2_dim, self.ninety_degree_2_dim), 1)
        self.assertEqual(manhattan(self.origin_2_dim, self.one_eighty_degree_2_dim), 1)
        self.assertEqual(manhattan(self.origin_2_dim, self.two_seventy_degree_2_dim), 1)

        # 90 Degree Rotations Around Origin Starting At 45 Degrees
        self.assertEqual(manhattan(self.origin_2_dim, self.forty_five_degree_2_dim), sqrt(2))
        self.assertEqual(manhattan(self.origin_2_dim, self.one_thirty_five_degree_2_dim), sqrt(2))
        self.assertEqual(manhattan(self.origin_2_dim, self.two_twenty_five_degree_2_dim), sqrt(2))
        self.assertEqual(manhattan(self.origin_2_dim, self.three_fifteen_degree_2_dim), sqrt(2))

    def test_cosine(self):
        # Undefined With A Vector Of All Zeros
        self.assertTrue(np.isnan(cosine(self.origin_2_dim, self.origin_2_dim)))
        self.assertTrue(np.isnan(cosine(self.origin_2_dim, self.zero_degree_2_dim)))
        self.assertTrue(np.isnan(cosine(self.zero_degree_2_dim, self.origin_2_dim)))

        # 90 Degree Rotations Around Origin
        self.assertAlmostEqual(cosine(self.zero_degree_2_dim, self.zero_degree_2_dim), 1-cos(0))
        self.assertAlmostEqual(cosine(self.zero_degree_2_dim, self.ninety_degree_2_dim), 1-cos(pi/2))
        self.assertAlmostEqual(cosine(self.zero_degree_2_dim, self.one_eighty_degree_2_dim), 1-cos(pi))
        self.assertAlmostEqual(cosine(self.zero_degree_2_dim, self.two_seventy_degree_2_dim), 1-cos(3*pi/2))

        # 90 Degree Rotations Around Origin Starting At 45 Degrees
        self.assertAlmostEqual(cosine(self.zero_degree_2_dim, self.forty_five_degree_2_dim), 1-cos(pi/4))
        self.assertAlmostEqual(cosine(self.zero_degree_2_dim, self.one_thirty_five_degree_2_dim), 1-cos(3*pi/4))
        self.assertAlmostEqual(cosine(self.zero_degree_2_dim, self.two_twenty_five_degree_2_dim), 1-cos(5*pi/4))
        self.assertAlmostEqual(cosine(self.zero_degree_2_dim, self.three_fifteen_degree_2_dim), 1-cos(7*pi/4))

    def test_chisqr(self):
        # Vector of Zeros
        self.assertTrue(np.isnan(chisqr(self.origin_2_dim, self.origin_2_dim)))

        # Example (Hand Calculated Result)
        self.assertEqual(chisqr(self.chi_vector_1, self.chi_vector_2), 0.12171612389003691)

    def test_pearson(self):
        # 0 Correlation Example
        self.assertEqual(pearson(self.zero_degree_2_dim, self.forty_five_degree_2_dim), 1)

        # 90 Degree Rotations Around Origin
        self.assertAlmostEqual(pearson(self.zero_degree_2_dim, self.zero_degree_2_dim), 0)
        self.assertAlmostEqual(pearson(self.zero_degree_2_dim, self.ninety_degree_2_dim), 2)
        self.assertAlmostEqual(pearson(self.zero_degree_2_dim, self.one_eighty_degree_2_dim), 2)
        self.assertAlmostEqual(pearson(self.zero_degree_2_dim, self.two_seventy_degree_2_dim), 0)

        # 90 Degree Rotations Around Origin Starting At 45 Degrees
        self.assertEqual(pearson(self.zero_degree_2_dim, self.forty_five_degree_2_dim), 1)
        self.assertEqual(pearson(self.zero_degree_2_dim, self.one_thirty_five_degree_2_dim), 2)
        self.assertEqual(pearson(self.zero_degree_2_dim, self.two_twenty_five_degree_2_dim), 1)
        self.assertEqual(pearson(self.zero_degree_2_dim, self.three_fifteen_degree_2_dim), 0)

    def test_hamming(self):
        self.assertEqual(hamming(self.bs_all_1_len_6, self.bs_all_0_len_6), 6)
        self.assertEqual(hamming(self.bs_all_1_len_6, self.bs_1_on_5_off), 5)
        self.assertEqual(hamming(self.bs_all_1_len_6, self.bs_2_on_4_off), 4)
        self.assertEqual(hamming(self.bs_all_1_len_6, self.bs_3_on_3_off), 3)
        self.assertEqual(hamming(self.bs_all_1_len_6, self.bs_4_on_2_off), 2)
        self.assertEqual(hamming(self.bs_all_1_len_6, self.bs_5_on_1_off), 1)
        self.assertEqual(hamming(self.bs_all_1_len_6, self.bs_all_1_len_6), 0)

if __name__ == '__main__':
    unittest.main()
