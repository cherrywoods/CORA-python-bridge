from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BenchmarkConfig:
    """Parsed benchmark configuration from YAML."""

    num_vars: int
    num_nn_input: int
    num_nn_output: int
    steps: int
    step_size: float

    # Variable info
    all_var_names: list[str]
    plant_state_names: list[str]
    nn_output_names: list[str]

    # Mappings: variable name -> 1-indexed MATLAB index
    var_to_x_index: dict[str, int]
    var_to_u_index: dict[str, int]

    # Initial set bounds (plant states only)
    initial_lb: list[float]
    initial_ub: list[float]

    # Dynamics expressions (plant states only, with original variable names)
    plant_dynamics: list[str]

    # Safety constraints (raw expressions)
    constraints_safe: list[str]

    # Model path
    model_path: str

    # Derived
    t_final: float = 0.0

    # Optional CORA options from YAML
    cora_options: dict = field(default_factory=dict)


def parse_config(yaml_path: str | Path) -> BenchmarkConfig:
    """Parse a benchmark YAML config file."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    num_vars = raw["num_vars"]
    num_nn_input = raw["num_nn_input"]
    num_nn_output = raw["num_nn_output"]
    steps = raw["steps"]
    step_size = raw["step_size"]

    initial_set = raw["initial_set"]
    dynamics_expressions = raw["dynamics_expressions"]
    constraints_safe = raw.get("constraints_safe", [])

    all_var_names = [v["name"] for v in initial_set]
    if len(all_var_names) != num_vars:
        raise ValueError(
            f"initial_set has {len(all_var_names)} vars but num_vars={num_vars}"
        )
    if len(dynamics_expressions) != num_vars:
        raise ValueError(
            f"dynamics_expressions has {len(dynamics_expressions)} entries "
            f"but num_vars={num_vars}"
        )

    # First num_nn_input variables are plant states -> x(1)..x(n)
    plant_state_names = all_var_names[:num_nn_input]
    var_to_x_index = {name: i + 1 for i, name in enumerate(plant_state_names)}

    # Remaining variables: identify time ("1" dynamics) and NN outputs ("0" dynamics)
    nn_output_names = []
    for i in range(num_nn_input, num_vars):
        expr = str(dynamics_expressions[i]).strip()
        if expr == "1":
            # Time variable, skip
            continue
        elif expr == "0":
            nn_output_names.append(all_var_names[i])
        else:
            raise ValueError(
                f"Unexpected dynamics expression '{expr}' for non-plant variable "
                f"'{all_var_names[i]}' (expected '0' or '1')"
            )

    if len(nn_output_names) != num_nn_output:
        raise ValueError(
            f"Found {len(nn_output_names)} NN output variables "
            f"but num_nn_output={num_nn_output}"
        )

    var_to_u_index = {name: i + 1 for i, name in enumerate(nn_output_names)}

    # Initial set bounds (plant states only)
    initial_lb = [float(initial_set[i]["interval"][0]) for i in range(num_nn_input)]
    initial_ub = [float(initial_set[i]["interval"][1]) for i in range(num_nn_input)]

    # Plant dynamics (first num_nn_input expressions)
    plant_dynamics = [str(dynamics_expressions[i]) for i in range(num_nn_input)]

    # Model path resolution
    model_dir = raw.get("model_dir", "")
    if model_dir:
        model_path = Path(model_dir)
        if not model_path.is_absolute():
            model_path = (yaml_path.parent / model_path).resolve()
        model_path = str(model_path)
    else:
        model_path = ""

    # CORA options from YAML
    cora_options = raw.get("cora_options", {})

    return BenchmarkConfig(
        num_vars=num_vars,
        num_nn_input=num_nn_input,
        num_nn_output=num_nn_output,
        steps=steps,
        step_size=step_size,
        all_var_names=all_var_names,
        plant_state_names=plant_state_names,
        nn_output_names=nn_output_names,
        var_to_x_index=var_to_x_index,
        var_to_u_index=var_to_u_index,
        initial_lb=initial_lb,
        initial_ub=initial_ub,
        plant_dynamics=plant_dynamics,
        constraints_safe=constraints_safe,
        model_path=model_path,
        t_final=steps * step_size,
        cora_options=cora_options,
    )
