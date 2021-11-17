#required to compile cython. normally no need to change it.

import os
import sys
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext


#standard
ext_modules = [
  Extension(
    "wigner_sums",
    ["wigner_sums.pyx"],
    language='c++',
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
  ),
]


setup(
  name = 'driver cython app',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
