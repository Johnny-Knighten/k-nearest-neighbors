from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["knn/ball_tree.pyx",
                           "knn/distance_metrics.pyx"]),
    include_dirs=[numpy.get_include()]
)
