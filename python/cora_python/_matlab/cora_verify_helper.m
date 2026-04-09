function [res, elapsed, traj_t, traj_x, traj_u, rt_lb, rt_ub, rt_t] = cora_verify_helper( ...
    dynamics_name, num_states, num_inputs, ...
    R0_lb, R0_ub, safe_lb, safe_ub, ...
    tFinal, samplingTime, onnx_path, ...
    reachTimeStep, tensorOrder, taylorTerms, zonotopeOrder, poly_method)
% cora_verify_helper - Helper function called from Python to run CORA
%    verification on a neural network controlled system.
%
% Inputs:
%    dynamics_name  - name of the dynamics function (must be on MATLAB path)
%    num_states     - number of plant states
%    num_inputs     - number of NN controller outputs
%    R0_lb, R0_ub   - initial set bounds (column vectors)
%    safe_lb, safe_ub - safe set bounds (column vectors)
%    tFinal         - final time
%    samplingTime   - NN controller sampling period (actuation step size)
%    onnx_path      - path to ONNX controller file
%    reachTimeStep  - time step for reachability analysis (must divide samplingTime)
%    tensorOrder    - tensor order for Taylor expansion
%    taylorTerms    - number of Taylor terms
%    zonotopeOrder  - zonotope order
%    poly_method    - NN evaluation method (e.g., "singh")
%
% Outputs:
%    res     - verification result string
%    elapsed - elapsed time in seconds
%    traj_t  - counterexample time points (empty if not FALSIFIED)
%    traj_x  - counterexample states (empty if not FALSIFIED)
%    traj_u  - counterexample NN outputs (empty if not FALSIFIED)
%    rt_lb   - reachtube lower bounds (num_states x N, empty if FALSIFIED)
%    rt_ub   - reachtube upper bounds (num_states x N, empty if FALSIFIED)
%    rt_t    - reachtube time points (1 x N, empty if FALSIFIED)

% ------------------------------ BEGIN CODE -------------------------------

% Ensure column vectors
R0_lb = R0_lb(:);
R0_ub = R0_ub(:);
safe_lb = safe_lb(:);
safe_ub = safe_ub(:);

% System Dynamics
fun = str2func(dynamics_name);
sys = nonlinearSys(fun, num_states, num_inputs);

% Neural Network Controller
nn = neuralNetwork.readONNXNetwork(onnx_path);

% Neural Network Controlled System
sys = neurNetContrSys(sys, nn, samplingTime);

% Parameters
params.tFinal = tFinal;
params.R0 = zonotope(interval(R0_lb, R0_ub));

% Reachability Options
options.timeStep = reachTimeStep;
options.alg = 'lin';
options.tensorOrder = tensorOrder;
options.taylorTerms = taylorTerms;
options.zonotopeOrder = zonotopeOrder;

% NN Evaluation Options
options.nn = struct();
options.nn.poly_method = poly_method;

% Specification
spec = specification(interval(safe_lb, safe_ub), 'safeSet');

% Verification
t_start = tic;
[res, R, traj] = verify(sys, spec, params, options, true);
elapsed = toc(t_start);

% Extract counterexample trajectory
traj_t = [];
traj_x = [];
traj_u = [];
if strcmp(res, 'FALSIFIED') && ~isempty(traj)
    traj_t = traj.t;
    traj_x = traj.x;

    % CORA does not store NN outputs in the trajectory, so re-evaluate
    % the network at each control step to recover them.
    n_points = size(traj_x, 2);
    n_steps = floor(traj_t(end) / samplingTime);
    points_per_step = floor(n_points / n_steps);
    traj_u = zeros(num_inputs, n_points);
    for k = 1:n_steps
        idx_start = (k-1) * points_per_step + 1;
        if k < n_steps
            idx_end = k * points_per_step;
        else
            idx_end = n_points;
        end
        % NN input is the state at the start of the control step
        nn_input = traj_x(:, idx_start);
        nn_output = nn.evaluate(nn_input);
        traj_u(:, idx_start:idx_end) = repmat(nn_output, 1, idx_end - idx_start + 1);
    end
end

% Extract reachtube interval bounds for VERIFIED / UNKNOWN
rt_lb = [];
rt_ub = [];
rt_t = [];
if ~strcmp(res, 'FALSIFIED') && ~isempty(R)
    % Collect time-interval sets across all reachSet array entries.
    % Each R(j) corresponds to one control step and contains
    % timeInterval.set{1} (the overapproximation over that interval).
    % Some entries may have empty timeInterval (e.g. after early abort),
    % so we skip those.
    n_segments = numel(R);
    tmp_lb = zeros(num_states, n_segments);
    tmp_ub = zeros(num_states, n_segments);
    tmp_t = zeros(1, n_segments);
    count = 0;
    for j = 1:n_segments
        if isempty(R(j).timeInterval) || isempty(R(j).timeInterval.set)
            continue;
        end
        count = count + 1;
        I = interval(R(j).timeInterval.set{1});
        tmp_lb(:, count) = infimum(I);
        tmp_ub(:, count) = supremum(I);
        % Time at the end of this interval
        t_interval = R(j).timeInterval.time{1};
        if isa(t_interval, 'interval')
            tmp_t(count) = supremum(t_interval);
        else
            tmp_t(count) = t_interval;
        end
    end
    if count > 0
        rt_lb = tmp_lb(:, 1:count);
        rt_ub = tmp_ub(:, 1:count);
        rt_t = tmp_t(1:count);
    end
end

% ------------------------------ END OF CODE ------------------------------
