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
class VirtualCounterexamples:
    """Support-function witnesses on the reachtube's segment sets, plus
    matching starting points in R_0.

    Each row of ``x`` is a point on a CORA reach-set zonotope (or
    polyZonotope) that maximises one signed axis-aligned direction at
    one segment. Unlike the corners of the segment's interval hull,
    these points lie inside the actual reach set.

    ``x0`` carries one starting point per axis direction -- the
    supportFunc vertex of R_0 in that direction. These are the
    initial states the consumer simulates forward from to construct
    virtual-CX (x0, oc_real) training pairs. ``x0_direction`` is a
    parallel array of signed 1-based dim indices.
    """

    x: np.ndarray  # (M, num_states) witness states
    t: np.ndarray  # (M,) witness times
    direction: np.ndarray  # (M,) signed 1-based dim index (s*i)
    x0: np.ndarray  # (D, num_states) R_0 starting points (D = 2 * num_states)
    x0_direction: np.ndarray  # (D,) signed 1-based dim index


@dataclass
class VerificationResult:
    """Result of CORA verification."""

    status: str  # "VERIFIED", "FALSIFIED", "UNKNOWN"
    time_seconds: float
    counterexample: CounterexampleTrace | None = None
    reachtube: Reachtube | None = None
    virtual_cxs: VirtualCounterexamples | None = None
