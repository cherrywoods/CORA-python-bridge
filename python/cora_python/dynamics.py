import hashlib
import re
import tempfile
from pathlib import Path

from .config import BenchmarkConfig


def _substitute_variables(
    expr: str,
    var_to_x: dict[str, int],
    var_to_u: dict[str, int],
) -> str:
    """Replace named variables with MATLAB x(i)/u(j) indexing.

    Sorts by variable name length descending to avoid partial matches
    (e.g., "psi" is replaced before "p").
    Uses word-boundary regex for safe substitution.
    """
    replacements: list[tuple[str, str]] = []
    for name, idx in var_to_x.items():
        replacements.append((name, f"x({idx})"))
    for name, idx in var_to_u.items():
        replacements.append((name, f"u({idx})"))

    # Sort by name length descending
    replacements.sort(key=lambda r: len(r[0]), reverse=True)

    for name, matlab_ref in replacements:
        expr = re.sub(r"\b" + re.escape(name) + r"\b", matlab_ref, expr)

    return expr


def generate_dynamics_code(config: BenchmarkConfig) -> tuple[str, str]:
    """Generate MATLAB dynamics function code from config.

    Returns (matlab_code, function_name).
    """
    # Build substituted expressions
    matlab_exprs = []
    for expr in config.plant_dynamics:
        substituted = _substitute_variables(
            expr, config.var_to_x_index, config.var_to_u_index
        )
        matlab_exprs.append(substituted)

    # Create a stable function name based on content hash
    content_hash = hashlib.md5(
        ";".join(matlab_exprs).encode()
    ).hexdigest()[:12]
    func_name = f"dynamics_{content_hash}"

    # Generate MATLAB function
    expr_str = "; ...\n         ".join(matlab_exprs)
    code = f"""\
function f = {func_name}(x, u)
    f = [{expr_str}];
end
"""
    return code, func_name


def write_dynamics_file(
    config: BenchmarkConfig,
    output_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Write a MATLAB dynamics .m file for the benchmark.

    Returns (directory_path, function_name).
    The directory must be added to the MATLAB path.
    """
    code, func_name = generate_dynamics_code(config)

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="cora_dynamics_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"{func_name}.m"
    file_path.write_text(code)

    return str(output_dir), func_name
