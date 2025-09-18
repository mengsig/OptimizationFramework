"""
Alias package for OptimizationFramework so you can write:

    import OF
    from OF.optimizers import ...
    from OF.lossFunctions import ...
    from OF.plottingUtils import ...
"""

from __future__ import annotations
import sys as _sys
import OptimizationFramework as _OF

# Re-export package version (optional)
__version__ = getattr(_OF, "__version__", None)

# Re-export modules in the OF.* namespace
optimizers = _OF.optimizers
lossFunctions = _OF.lossFunctions  # lowercase view of LossFunctions
plottingUtils = _OF.plottingUtils  # lowercase view of PlotDefaults

# Ensure "from OF.lossFunctions import ..." works
_sys.modules[__name__ + ".optimizers"] = optimizers
_sys.modules[__name__ + ".lossFunctions"] = lossFunctions
_sys.modules[__name__ + ".plottingUtils"] = plottingUtils

# Also support original-cased names if someone uses them
SysLoss = getattr(_OF, "LossFunctions", None)
SysPlot = getattr(_OF, "PlotDefaults", None)
if SysLoss is not None:
    _sys.modules[__name__ + ".LossFunctions"] = SysLoss
if SysPlot is not None:
    _sys.modules[__name__ + ".PlotDefaults"] = SysPlot

__all__ = ["optimizers", "lossFunctions", "plottingUtils"]
