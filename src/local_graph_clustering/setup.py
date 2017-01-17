from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

print numpy.get_include()

ext_modules=[ Extension("affinity",
              ["affinity.pyx"],
              libraries=["m"],
              include_dirs=[numpy.get_include()],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "affinity",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)

ext_modules=[ Extension("acl",
              ["acl.pyx"],
              libraries=["m"],
              include_dirs=[numpy.get_include()],
              extra_compile_args = ["-O3", "-ffast-math"])]

setup(
  name = "acl",
  cmdclass = {"build_ext": build_ext},
  include_dirs=[numpy.get_include()],
  ext_modules = ext_modules)


ext_modules=[ Extension("l1_distance_matrix",
              ["l1_distance_matrix.pyx"],
              libraries=["m"],
              include_dirs=[numpy.get_include()],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "l1_distance_matrix",
  cmdclass = {"build_ext": build_ext},
  include_dirs=[numpy.get_include()],
  ext_modules = ext_modules)

ext_modules=[ Extension("KL_divergence",
              ["KL_divergence.pyx"],
              libraries=["m"],
              include_dirs=[numpy.get_include()],
              extra_compile_args = ["-O3", "-ffast-math" ])]

setup(
  name = "KL_divergence",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)