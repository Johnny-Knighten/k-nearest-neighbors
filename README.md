# K Nearest Neighbors

K Nearest Neighbors(KNN) is a simple machine learning algorithm that can
be used for both classification and regression. It is a lazy learning
algorithm, that is the generalization of the data occurs while querying
instead of during a training phase.

Given a collection of training data represented by N dimensional vectors
and a set of N dimensional testing/query vectors, find the K closest
vectors in the training data for each query vector. To do this we use
a metric that measures the distance between two vectors. We then use
the K closest neighbors to perform regression or classification for the
query vectors. The general idea is that the points that are closest to
the query vectors should be more similar than the points that are 
farther away.

For classification, we find the K nearest neighbors and use their 
assigned labels to vote for what the query vector label should be. For
instance, if we find the three nearest neighbors and their respective 
labels are A, A, B then we would label the query vector as A.

For regression, we find the K nearest neighbors and use their outputs to
predict the output for the query vector. To predict the output we will 
simply find the mean of the nearest neighbors' outputs. For instance,
if we find the three nearest neighbors and their respective outputs are
1, 3, 5 then we would predict the query points output to be 3.

There are numerous ways to perform a nearest neighbor search, but three
most popular methods are: brute force, using a special data structure,
and using approximate methods. In this project we only focus on two
methods, brute force and using a special data structure(Ball Tree).


## Brute Force

Brute force KNN is the simplest version to implement, but its the most
computationally expensive. In brute force KNN we calculate the distance
between every query point and every training point. This means the 
running time will grow with the number of training points, number of 
query points, and the dimensionality of the data(the calculation of
distance slows down as dimensionality increase).


## Ball Tree

A Ball Tree is a special data structure that attempts to partition the
training data in such a way that only a portion of the training data has
to be searched. Ball Tree's partition data into balls(hyperspheres). You
can imagine that you are able to place a circle(2dims)/sphere(3dims) 
around all of your training data. Then you can split the data contained
inside the circle/sphere and place 2 more circles/spheres. Then you keep
partitioning all circle/spheres until there are only circle/spheres that
contain a specified amount of points. This partitioning can be 
represented as a binary tree. The internal tree nodes represent the 
circles/spheres/hyperspheres and the leafs are collections of data 
points contained by the lowest level circles/spheres/hyperspheres. 

The idea is when a query is performed we use the tree to prune off paths
that cannot contain points that are closer than the ones discovered so
far. To achieve this we calculate the distance between the query point
and the center of the current circle/sphere/hypersphere. When then 
subtract the radius of the current circle/sphere/hypersphere to get the
distance from the query point to the edge of the 
circle/sphere/hypersphere. If the distance to the edge is further away 
than the farthest away point in the set of closest points discovered so
far, than that circle/sphere/hypersphere cannot contain a point that 
would be any closer than the ones discovered so far. If a 
circle/sphere/hypersphere cannot contain a point any closer than the 
ones discovered so far than we prune that path in the tree and continue
the search.

The efficiency of Ball Trees largely depends on the structure of the
training data. If the training data can naturally be partitioned into
circles/spheres/hyperspheres than the Ball Tree can be very efficient;
however, data with no real structure(white noise) do not perform well. 
Data that is structured can possibly cause more paths to be pruned, 
while unstructured data may lead to more paths being explored. For this 
reason, the curse of dimensionality impacts Ball Trees heavily. As the 
dimensionality increase, it is less likely that the data can be 
partitioned into nice circles/spheres/hyperspheres. As more paths are
explored the performance slowly begins to match brute force KNN.

In general Ball Trees performance depends: on the amount of training
data, the number of query points, the dimensionality of the data, and 
the structure of training data. Performance will grow linearly with the
number of query points. If there is structure in the training data
(usually meaning low dim) then performance grows logarithmically with
the number of training data. Ideally since the Ball Tree is a binary
tree only a few paths will have to be explored and the length of these
paths(the trees depth) will be logarithmic with the amount of training
data. Dimensionality will effect the performance of metric calculations
and influence the structure of the training data.

One last detail about Ball Trees is that the distance metric used must
follow the triangle inequality.


### Ball Tree Relevant Reading

https://en.wikipedia.org/wiki/Ball_tree
Five Balltree Construction Algorithms - STEPHEN M. OMOHUNDRO
ftp://ftp.icsi.berkeley.edu/pub/techreports/1989/tr-89-063.pdf
http://www.cs.cornell.edu/courses/cs4780/2017sp/lectures/lecturenote16.html


# Implementation Details

Originally all code was implemented using pure python along with numpy.
This caused a few issues. First, although numpy is fast, it can cause
heavy memory usage when numpy arrays are broadcasted(which happens when
calculating the pairwise distance between two collections of
vectors). Second, pure python implementations are slow. This was mainly
an issue with the pure python implementation of Ball Tree. Even
though numpy was still used for distance metrics, the pure python
implementation of Ball Tree was still slow.

To get around these issues Cython was used instead. Cython is used to
implement all distance metrics as well as the Ball Tree implementation.

The following metrics are available:

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

Prototypes of old implementations can be found in the jupyter notebooks
in the prototypes directory.


# How To Install

Yet To Be Completed.


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
3. Minimize duplicate code in k_nearest_neighbors.py