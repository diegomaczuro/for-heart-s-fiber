from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
#python setup.py build_ext --inplace

ext_modules=[
    Extension("petsc_2d",
              ["petsc_2d.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              )
]

setup(
  name = "petsc_2d",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)