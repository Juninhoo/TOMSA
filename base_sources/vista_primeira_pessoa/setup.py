# Compilar:  python setup.py build_ext -b python
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext = Pybind11Extension("uwmath", ["cpp/src/bindings.cpp"], include_dirs=["cpp/include"], cxx_std=17)
setup(name="uwmath", version="0.1", ext_modules=[ext], cmdclass={"build_ext": build_ext})
