import shutil
from pathlib import Path

import matlab
import numpy as np

from .config import BenchmarkConfig, parse_config
from .constraints import parse_box_constraints
from .dynamics import write_dynamics_file
from .matlab_bridge import MatlabBridge
from .types import (
    CounterexampleTrace,
    Reachtube,
    VerificationResult,
    VirtualCounterexamples,
)

# Default CORA options from ARCH-COMP examples
_DEFAULT_OPTIONS = {
    "reachTimeStep": 0.01,
    "tensorOrder": 2,
    "taylorTerms": 4,
    "zonotopeOrder": 20,
    "poly_method": "singh",
    # Max depth of CORA's recursive initial-set splitting branch-and-bound
    # (see @neurNetContrSys/verify.m). 0 disables splitting.
    "splitR0": 0,
    # Extract one supportFunc witness per axis-aligned direction per
    # reachtube segment. Off by default; turn on to get virtual
    # counterexamples that lie on CORA's actual reach set rather than on
    # the looser interval hull.
    "extract_virtual_cx": False,
    # supportFunc method used to compute the witness. 'upper' is
    # closed-form and sound; 'split' / 'conZonotope' / 'bnb' / etc. are
    # tighter but more expensive.
    "virtual_cx_method": "upper",
}


def verify_from_config(
    config_path: str | Path,
    onnx_path: str | Path | None = None,
    cora_options: dict | None = None,
    engine: MatlabBridge | None = None,
) -> VerificationResult:
    """Run CORA verification on a benchmark defined by a YAML config.

    Args:
        config_path: Path to the YAML benchmark config file.
        onnx_path: Optional override for the ONNX controller path.
            If None, uses the model_dir from the YAML config.
        cora_options: Optional dict of CORA reachability options.
            Keys: reachTimeStep, tensorOrder, taylorTerms, zonotopeOrder,
            poly_method, splitR0. Overrides both defaults and YAML
            cora_options.
        engine: Optional MatlabBridge instance to reuse.
            If None, a new engine is started and stopped after verification.

    Returns:
        VerificationResult with status, timing, and optional counterexample.
    """
    # Parse config
    config = parse_config(config_path)

    # Resolve ONNX path
    if onnx_path is not None:
        model_path = str(Path(onnx_path).resolve())
    else:
        model_path = config.model_path

    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not model_path.endswith(".onnx"):
        raise ValueError(
            f"Only .onnx models are supported, got: {model_path}. "
            f"Convert .pt models to .onnx first."
        )

    # Parse constraints
    safe_lb, safe_ub = parse_box_constraints(config)

    # Generate dynamics .m file
    dynamics_dir, func_name = write_dynamics_file(config)

    # Resolve CORA options: defaults <- YAML <- Python API
    opts = dict(_DEFAULT_OPTIONS)
    opts.update(config.cora_options)
    if cora_options:
        opts.update(cora_options)

    # Engine management
    owns_engine = engine is None
    if owns_engine:
        engine = MatlabBridge()
        engine.start()

    try:
        # Add dynamics directory to MATLAB path
        engine.add_to_path(dynamics_dir)

        # Convert arrays to MATLAB doubles
        R0_lb = matlab.double(config.initial_lb)
        R0_ub = matlab.double(config.initial_ub)
        m_safe_lb = matlab.double(safe_lb)
        m_safe_ub = matlab.double(safe_ub)

        # Call MATLAB helper
        (res, elapsed, traj_t, traj_x, traj_u, rt_lb, rt_ub, rt_t,
         vcx_x, vcx_t, vcx_dir, vcx_x0, vcx_x0_dir) = (
            engine.engine.cora_verify_helper(
                func_name,
                float(config.num_nn_input),
                float(config.num_nn_output),
                R0_lb,
                R0_ub,
                m_safe_lb,
                m_safe_ub,
                float(config.t_final),
                float(config.step_size),
                model_path,
                float(opts["reachTimeStep"]),
                float(opts["tensorOrder"]),
                float(opts["taylorTerms"]),
                float(opts["zonotopeOrder"]),
                opts["poly_method"],
                float(opts["splitR0"]),
                bool(opts["extract_virtual_cx"]),
                opts["virtual_cx_method"],
                nargout=13,
            )
        )

        # Parse counterexample
        counterexample = None
        if res == "FALSIFIED" and traj_t:
            # MATLAB returns (dim, T) arrays; transpose to (T, dim).
            # Use reshape(-1) instead of squeeze() so a single-timestamp
            # trace stays 1-D rather than collapsing to a 0-D scalar.
            counterexample = CounterexampleTrace(
                t=np.asarray(traj_t).reshape(-1),
                x=np.asarray(traj_x).T,
                u=np.asarray(traj_u).T,
            )

        # Parse reachtube (available for VERIFIED / UNKNOWN)
        reachtube = None
        if res != "FALSIFIED" and rt_lb:
            # MATLAB returns (num_states, N); transpose to (N, num_states).
            # reshape(-1) keeps a single-interval reach 1-D.
            reachtube = Reachtube(
                t=np.asarray(rt_t).reshape(-1),
                lb=np.asarray(rt_lb).T,
                ub=np.asarray(rt_ub).T,
            )

        # Parse virtual counterexamples (one per axis direction per
        # reachtube segment when extract_virtual_cx=True), plus the
        # supportFunc vertices of R_0 (one per axis direction).
        virtual_cxs = None
        if vcx_x:
            virtual_cxs = VirtualCounterexamples(
                x=np.asarray(vcx_x).T,
                t=np.asarray(vcx_t).reshape(-1),
                direction=np.asarray(vcx_dir).reshape(-1).astype(int),
                x0=np.asarray(vcx_x0).T,
                x0_direction=np.asarray(vcx_x0_dir).reshape(-1).astype(int),
            )

        return VerificationResult(
            status=res,
            time_seconds=float(elapsed),
            counterexample=counterexample,
            reachtube=reachtube,
            virtual_cxs=virtual_cxs,
        )

    finally:
        # Clean up temp dynamics directory
        shutil.rmtree(dynamics_dir, ignore_errors=True)

        if owns_engine:
            engine.stop()
