SimulationSupport documentation
================================

This repository contains Python routines that are shared between SpEC and
SpECTRE, such as eccentricity control.

Install the package and its optional dependencies into a virtual environment
like this:

.. code-block:: bash

   python -m pip install .[docs,dev]

The typical workflow is to move SpEC Python tools into this repository with
little to no changes, and then import the tools in both SpEC and SpECTRE:

1. Move the Python file from SpEC into this repository. Construct a sensible
   directory structure if needed. Add empty `__init__.py` files as necessary.

2. Remove any SpEC-specific dependencies. For example, remove imports from other
   SpEC modules by either inlining the imported functions or moving the
   dependencies into this repository as well. Change no logic or even output
   (like print statements), since SpEC may depend on the exact behavior.

3. Add a small note at the top of the file indicating where the file came from
   and what changes were made (if any).

4. Commit, push, and open a pull request. Once the pull request is merged,
   update the SimulationSupport version hash used by SpEC and update SpEC to
   import the file from SimulationSupport instead of the local copy. This way,
   no changes are needed in SpEC.

5. Modernize the code over time, making sure that both SpEC and SpECTRE remain
   compatible. Unit tests in both SpEC and SpECTRE help ensure this.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   Workflow
   Api
