from dataclasses import dataclass

import numpy as np


@dataclass
class CounterexampleTrace:
    """Trajectory data from a falsifying simulation."""

    t: np.ndarray  # (T,) time points
    x: np.ndarray  # (T, num_states) plant states
    u: np.ndarray  # (T, num_inputs) NN controller outputs


@dataclass
class VerificationResult:
    """Result of CORA verification."""

    status: str  # "VERIFIED", "FALSIFIED", "UNKNOWN"
    time_seconds: float
    counterexample: CounterexampleTrace | None = None
