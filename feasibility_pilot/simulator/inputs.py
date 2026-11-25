"""Compatibility imports for the pilot's original simulator.inputs module."""

from simulator.workload import arrivals, rng

if __package__ == "simulator":  # historical working directory
    from experiments.workload import inputs
else:
    from ..experiments.workload import inputs

__all__ = ["arrivals", "inputs", "rng"]
