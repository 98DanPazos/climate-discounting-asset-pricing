"""
climate_discount — a small, tested engine for long-horizon discounting.

Modules
-------
paths        : locate the repo root and standard folders
discounting  : the four discounting methods (constant, gamma, Newell-Pizer,
               consumption-based SDF) returning a common DiscountSchedule
calibrate    : estimate the methods' parameters from the project's data
"""
from .discounting import (
    DiscountSchedule,
    constant,
    gamma,
    newell_pizer,
    ramsey_sdf,
    ramsey_sdf_montecarlo,
)

__all__ = [
    "DiscountSchedule", "constant", "gamma", "newell_pizer",
    "ramsey_sdf", "ramsey_sdf_montecarlo",
]
