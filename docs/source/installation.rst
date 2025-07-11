Installation
============

You can install **Infinity Tomographic Reconstruction** from source using `setuptools`.
(Still in test)

----

Install Directly from GitHub
----------------------------

You can install directly from the GitHub repository without cloning:

.. code-block:: bash

   pip install git+https://github.com/DRIM-UA/infinity-tomographic-reconstruction.git

This will fetch the latest `main` branch and install it using `setuptools`.

----


Install from Local Source
-------------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/crispim-enc/Infinity-Tomographic-Reconstruction.git
   cd infinity-tomographic-reconstruction

2. Install the project using `setuptools`:

.. code-block:: bash

   pip install .

For development, use **editable mode**:

.. code-block:: bash

   pip install -e .

This allows live updates to your code without re-installing.

----


Pre-Requirements
------------

    Cuda Toolkit 11.3 or higher,
    Python 3.8, pip, cl compiler.


Core dependencies (automatically installed with `pip install`) include:

- `numpy>=2.0`
- `matplotlib`
- `scipy`
- `pycuda`
-  "vtk",
- "pydicom"
- "pycuda"
- "json5",
- "scikit-image"
- "pandas"
- "PyQt6"
- "uproot"

To install extra dev tools:

.. code-block:: bash

   pip install -e .[dev]

----

That's it! You're ready to set your reconstruction modules 🚀
