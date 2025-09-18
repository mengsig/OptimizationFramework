"""
OptimizationFramework
Convenient, tidy imports with lowercase submodule names:

    from OptimizationFramework.optimizers import ...
    from OptimizationFramework.lossFunctions import ...
    from OptimizationFramework.plottingUtils import ...

The actual source files are:
    - optimizers.py
    - LossFunctions.py
    - plottingUtils.py
"""

from __future__ import annotations
import sys as _sys

__all__ = [
    "optimizers",
    "lossFunctions",
    "plottingUtils",
    # also expose the original-cased modules for completeness:
    "lossFunctions",
    "plottingUtils",
]

# Import the real modules
from . import optimizers as optimizers  # real file: optimizers.py
from . import lossFunctions as LossFunctions  # real file: LossFunctions.py
from . import plottingUtils as PlottingUtils  # real file: plottingUtils.py

# Provide lowercase-friendly aliases as submodules
lossFunctions = LossFunctions
plottingUtils = PlottingUtils

# Make them importable as OptimizationFramework.lossFunctions / plottingUtils
_sys.modules[__name__ + ".lossFunctions"] = LossFunctions
_sys.modules[__name__ + ".plottingUtils"] = PlottingUtils

# (Optional) expose version here if you like
__version__ = "0.1.0"
