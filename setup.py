from distutils.core import setup
import numpy
from glob import glob
import os


try:
    from Cython.Distutils.extension import Extension
    from Cython.Distutils import build_ext
except ImportError:
    from setuptools import Extension
    USING_CYTHON = False
else:
    USING_CYTHON = True


ext = 'pyx' if USING_CYTHON else 'c'
sources = glob('knn/*.%s' % (ext,))
extensions = [
    Extension(source.split('.')[0].replace(os.path.sep, '.'),
              sources=[source],)
    for source in sources]
cmdclass = {'build_ext': build_ext} if USING_CYTHON else {}


setup(
    name='k-nearest-neighbors',
    version='0.1',
    description='KNN using brute force and ball trees implemented in Python/Cython',
    url='https://github.com/JKnighten/k-nearest-neighbors',
    author='Jonathan Knighten',
    author_email='jknigh28@gmail.com',
    ext_modules=extensions,
    cmdclass=cmdclass,
    include_dirs=[numpy.get_include()],
    setup_requires=['numpy'],
    zip_safe=False
)
