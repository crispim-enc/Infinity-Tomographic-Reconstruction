#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: setup
# * AUTHOR: Pedro Encarnação
# * DATE: 09/01/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

from setuptools import setup, find_packages, Extension

from setuptools.command.build_ext import build_ext


class get_numpy_include(object):
    def __str__(self):
        import numpy
        return numpy.get_include()

ext_modules = [
    Extension(
        "your_module",
        ["your_module.cpp"],
        include_dirs=[get_numpy_include()],
    )
]

setup(
    name="Infinity-Tomographic-Reconstruction",               # Package name
    version="0.1.0",                      # Package version
    author="Pedro Encarnação",                   # Author's name
    author_email= "", # Author's email
    description="Tomographic reconstruction for diferent imaging modalities (PET, SPECT, CT and Compton) \n Image Reconstruction Framework for PET/CT and SPECT", # Short description
    # Detailed description (usually README)
    long_description_content_type="text/markdown", # Content type for long description
    url="https://github.com/DRIM-UA/Infinity-Tomographic-Reconstruction", # Project URL
    install_requires=[                    # Dependencies required for the project
        "numpy>=2.0.2",
    ],
    classifiers=[                         # Metadata for PyPI and search filters
        "Programming Language :: Python :: 3",
        "License :: CC BY-NC-SA 4.0",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),# Directory where the packages are located
    package_dir={"": "src"}# Minimum Python version required
)