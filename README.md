This is a Python/Cython implementation of KNN algorithms. Two algorithms are provided: a brute force algorithm 
implemented with numpy and a ball tree implemented using Cython. Also provided is a set of distance metrics that are 
implemented in Cython.

An overview of KNN and ball tress can be found [here](https://github.com/JKnighten/k-nearest-neighbors/wiki/KNN-and-BallTree-Overview).

# Distance Metrics Provided

Ball tress have less metrics available because the metrics that are used by the ball tree must follow the triangle
 inequality.

Brute Force
* Manhattan
* Euclidean
* Hamming
* Pearson
* Cosine
* Chi-Squared


Ball Tree
* Manhattan
* Euclidean
* Hamming

# How To Install

To install this package first ensure that pip is installed and
updated. Then execute the following code while in the package's main
directory(the directory containing setup.py):
```
 pip install .
```

This will install knn into your current python environment. You then
will import knn.models to access classes for classification and 
regression. knn.models also contains a mixin that provides nearest
neighbor search methods you can use in your own classes.

Note - Cython and Numpy are used in this project. Numpy is required but
Cython is not. If Cython is not installed the packaged .c 
files will be complied using your systems default c compiler. Numpy
will be installed when installing knn.

Note - This code was designed and tested using Python 3.5.2.



# Notes About Use

Currently all implemented algorithms expect numpy arrays as input. All 
numpy arrays are expected to be of type np.float.

This may be changed in the future.


# Example Use

Example use of the the implemented algorithms can be found in
example_usage.py.


# Possible Updates
1. Allow the use of numpy arrays of types other than np.float
2. Implemented Brute Force KNN in Cython
