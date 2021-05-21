from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

modules = [Extension("Cython_Ising",
                     ["Cython_Ising.pyx"]),
           Extension("Cython_Potts", ["Cython_Potts.pyx"])]

setup(name='Ising/Potts Model',
      ext_modules=cythonize(modules, annotate=True),
      include_dirs=[numpy.get_include()])
