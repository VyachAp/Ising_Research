from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

modules = [Extension("Cython_Ising",
                     ["Cython_Ising.pyx"])]

setup(name='Ising Model',
      ext_modules=cythonize(modules, annotate=True),
      include_dirs=[numpy.get_include()])
