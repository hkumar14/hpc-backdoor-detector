"""hpcdet: one-class detection of neural-network backdoor inferences from HPC traces."""
from .pipeline import HPCBackdoorDetector  # noqa: F401

__all__ = ["HPCBackdoorDetector"]
__version__ = "1.0.0"
