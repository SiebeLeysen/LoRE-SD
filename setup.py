import os

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

cpp_args = ['-std=c++11']

nlopt_dir = os.environ.get('NLOPT_DIR')
nlopt_include_dir = os.environ.get('NLOPT_INCLUDE_DIR') or (
    os.path.join(nlopt_dir, 'include') if nlopt_dir else None
)
nlopt_lib_dir = os.environ.get('NLOPT_LIB_DIR') or (
    os.path.join(nlopt_dir, 'lib') if nlopt_dir else None
)

include_dirs = [d for d in (nlopt_include_dir,) if d]
library_dirs = [d for d in (nlopt_lib_dir,) if d]


ext_modules = [
    Pybind11Extension(
        'lore_sd.cpp_optimiser',
        sources=[
            'src/lore_sd/cpp/bindings.cpp',
            'src/lore_sd/cpp/optimise.cpp',
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=['nlopt'],
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='lore_sd',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'pandas',
        'pybind11',
        'nlopt'
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'lore_dwi2decomposition=lore_sd.dwi2decomposition:main',
            'lore_angular_correlation=lore_sd.angular_correlation:main',
            'lore_decomposition2contrast=lore_sd.decomposition2contrast:main',
        ],
    },
    author='Siebe Leysen',
    author_email='siebe.leysen@kuleuven.be',
    description='A package for DWI decomposition: LoRE-SD decomposes the dMRI data into voxel-level ODFs and response functions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SiebeLeysen/LoRE-SD',
    python_requires='>=3.6',
)
