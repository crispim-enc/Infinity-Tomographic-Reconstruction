#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: setup
# * AUTHOR: Pedro Encarnação
# * DATE: 09/01/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

from setuptools import setup, find_packages

setup(
    name="Infinity-Tomographic-Reconstruction",               # Package name
    version="0.0.1",                      # Package version
    author="Pedro Encarnação",                   # Author's name
    author_email="pedro.encarnacao@ua.pt",# Author's email
    description="Tomographic reconstruction for diferent imaging modalities (PET, SPECT, CT and Compton) \n Image Reconstruction Framework for PET/CT and SPECT", # Short description
    long_description=open("README.md").read(), # Detailed description (usually README)
    long_description_content_type="text/markdown", # Content type for long description
    url="https://github.com/DRIM-UA/Infinity-Tomographic-Reconstruction", # Project URL
    packages=find_packages(),             # Automatically discover packages in the project
    install_requires=[                    # Dependencies required for the project
        "numpy>=1.21.0",
        "pandas>=1.3.0"
    ],
    classifiers=[                         # Metadata for PyPI and search filters
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: CC BY-NC-SA 4.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",              # Minimum Python version required
)