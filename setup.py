from distutils.core import setup, Extension
setup(name = 'pathfinder', version = '1.0', \
      ext_modules = [Extension('pathfinder', ['pathfindermodule.c'])])


