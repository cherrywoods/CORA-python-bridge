import math
import re

from .config import BenchmarkConfig


def parse_box_constraints(
    config: BenchmarkConfig,
) -> tuple[list[float], list[float]]:
    """Parse safety constraints into lower/upper bound arrays.

    Returns (lb, ub) arrays of length num_nn_input (plant state dimension).
    Raises ValueError if any constraint is nonlinear.
    """
    n = config.num_nn_input
    lb = [-math.inf] * n
    ub = [math.inf] * n

    plant_vars = set(config.plant_state_names)

    for expr_str in config.constraints_safe:
        expr = expr_str.strip()
        bound = _parse_single_box_constraint(expr, plant_vars)
        if bound is None:
            raise ValueError(
                f"Nonlinear constraint not supported: '{expr_str}'. "
                f"Only box constraints (e.g., '-x - 1.5', 'x - 2') are supported."
            )

        var_name, is_lower, value = bound
        idx = config.var_to_x_index[var_name] - 1  # 0-indexed for array

        if is_lower:
            lb[idx] = max(lb[idx], value)
        else:
            ub[idx] = min(ub[idx], value)

    return lb, ub


def _parse_single_box_constraint(
    expr: str,
    valid_vars: set[str],
) -> tuple[str, bool, float] | None:
    """Try to parse a single expression as a box constraint.

    Constraint format: expr <= 0.

    Returns (var_name, is_lower_bound, bound_value) or None if not a box constraint.

    Examples:
        "-th1 - 1.7"  -> th1 >= -1.7  -> ("th1", True, -1.7)
        "th1 - 2"     -> th1 <= 2     -> ("th1", False, 2.0)
        "-y - 1"      -> y >= -1      -> ("y", True, -1.0)
        "y - 1"       -> y <= 1       -> ("y", False, 1.0)
    """
    # Pattern: optional minus, variable name, then +/- constant
    # "-var - const" means -var - const <= 0 => var >= -const
    # "var - const" means var - const <= 0 => var <= const
    # "-var + const" means -var + const <= 0 => var >= const
    # "var + const" means var + const <= 0 => var <= -const
    pattern = r"^(-?)\s*(\w+)\s*([+-])\s*([\d.]+(?:e[+-]?\d+)?)$"
    m = re.match(pattern, expr.strip(), re.IGNORECASE)
    if m is None:
        return None

    negated = m.group(1) == "-"
    var_name = m.group(2)
    op = m.group(3)
    const = float(m.group(4))

    if var_name not in valid_vars:
        return None

    # Compute the bound value
    # The constraint is: (sign * var) + (op_sign * const) <= 0
    if negated:
        # -var +/- const <= 0 => var >= +/- const
        if op == "-":
            # -var - const <= 0 => var >= -const
            return (var_name, True, -const)
        else:
            # -var + const <= 0 => var >= const
            return (var_name, True, const)
    else:
        # var +/- const <= 0 => var <= -/+ const
        if op == "-":
            # var - const <= 0 => var <= const
            return (var_name, False, const)
        else:
            # var + const <= 0 => var <= -const
            return (var_name, False, -const)
