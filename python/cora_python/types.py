from dataclasses import dataclass

import numpy as np


@dataclass
class CounterexampleTrace:
    """Trajectory data from a falsifying simulation."""

    t: np.ndarray  # (T,) time points
    x: np.ndarray  # (T, num_states) plant states
    u: np.ndarray  # (T, num_inputs) NN controller outputs


@dataclass
class Reachtube:
    """Interval bounds of the reachable set over time (plant states only).

    Each column corresponds to one control step's time-interval
    overapproximation.
    """

    t: np.ndarray  # (N,) time points (end of each interval)
    lb: np.ndarray  # (N, num_states) lower bounds
    ub: np.ndarray  # (N, num_states) upper bounds


@dataclass
class VerificationResult:
    """Result of CORA verification."""

    status: str  # "VERIFIED", "FALSIFIED", "UNKNOWN"
    time_seconds: float
    counterexample: CounterexampleTrace | None = None
    reachtube: Reachtube | None = None
