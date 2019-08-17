from distutils.core import setup, Extension
import numpy

setup(name='pathfinder', version='1.0',
      ext_modules=[
          Extension(
              'pathfinder',
              ['src/pathfindermodule.cpp'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-std=c++17'],
          )
      ])
