project(
  'moist_euler_dg',
  ['c', 'fortran'],
  version: '1.0',
  license: 'MIT',
  meson_version: '>=1.3.0',
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

fc = meson.get_compiler('fortran')
fortran_args = fc.get_supported_arguments(
  '-fno-range-check',
  '-Wuninitialized',
)

if get_option('fast-math')
  fortran_args += fc.get_supported_arguments('-ffast-math')
endif

if get_option('native')
  fortran_args += fc.get_supported_arguments('-march=native')
endif

if fortran_args.length() > 0
  add_project_arguments(fortran_args, language: 'fortran')
endif

incdir_numpy = run_command(
  py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true,
).stdout().strip()

incdir_f2py = run_command(
  py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true,
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

fortran_sources = files(
  'moist_euler_dg/three_phase_thermo.F90',
  'moist_euler_dg/two_phase_thermo.F90',
  'moist_euler_dg/moist_euler_dynamics_2D.F90',
)

f2py_generated = custom_target(
  '_moist_euler_dg-f2py-generated',
  input: fortran_sources,
  output: [
    '_moist_euler_dgmodule.c',
    '_moist_euler_dg-f2pywrappers2.f90',
  ],
  command: [
    py,
    '-m',
    'numpy.f2py',
    '@INPUT@',
    '-m',
    '_moist_euler_dg',
    '--lower',
    '--verbose',
  ],
)

py.extension_module(
  '_moist_euler_dg',
  [
    fortran_sources,
    f2py_generated,
    incdir_f2py / 'fortranobject.c',
  ],
  include_directories: inc_np,
  dependencies: py_dep,
  install: true,
  install_dir: py.get_install_dir(),
)

py.install_sources(
  [
    'moist_euler_dg/__init__.py',
    'moist_euler_dg/euler_2D.py',
    'moist_euler_dg/fortran_three_phase_euler_2D.py',
    'moist_euler_dg/fortran_two_phase_euler_2D.py',
    'moist_euler_dg/non_equilibrium_euler_2D.py',
    'moist_euler_dg/one_phase_euler_2D.py',
    'moist_euler_dg/three_phase_euler_2D.py',
    'moist_euler_dg/two_phase_euler_2D.py',
    'moist_euler_dg/unstable_two_phase_euler_2D.py',
    'moist_euler_dg/utils.py',
  ],
  subdir: 'moist_euler_dg',
)
