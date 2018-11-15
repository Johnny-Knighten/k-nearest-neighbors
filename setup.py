from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["knn/non_broadcast_distance_metrics.pyx",
                           "knn/ball_tree.pyx",
                           "knn/distance_metrics_cython.pyx"]),
    include_dirs=[numpy.get_include()]
)
