import sys, types

# Create a fake distutils.msvccompiler module with a dummy get_build_version
msvccompiler_stub = types.ModuleType("distutils.msvccompiler")

def get_build_version():
    # Return a dummy version number; it is never actually used on Linux
    return None

msvccompiler_stub.get_build_version = get_build_version

sys.modules["distutils.msvccompiler"] = msvccompiler_stub

from distutils.core import setup
import site
from numpy.distutils.core import setup, Extension

site.ENABLE_USER_SITE = True

sources = [
    "./moist_euler_dg/three_phase_thermo.F90",
"./moist_euler_dg/two_phase_thermo.F90",
    "./moist_euler_dg/moist_euler_dynamics_2D.F90",
]

gnu_f90flags = ['-fno-range-check', '-march=native', '-ffast-math', '-fopenmp', '-Wuninitialized']

setup(
    name='moist_euler_dg',
    version='1.0',
    packages=['moist_euler_dg'],
    install_requires=["numpy", "matplotlib"],
    url='',
    license='',
    author='Kieran Ricardo',
    author_email='',
    description='',
    ext_modules=[
        Extension(name="_moist_euler_dg",
                sources=sources,
                extra_f90_compile_args=gnu_f90flags,
                f2py_options=['--verbose'],
                ),
    ]
)