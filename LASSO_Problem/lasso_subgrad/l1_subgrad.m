% Input information: model parameters A, b, mu, initial value of iteration x0, structure 'opts' containing algorithm parameters
% Output information: The iterative solution x (original LASSO problem) and a structure 'out' containing the iteration information

% out.fvec : the objective function value of the original LASSO problem at each iteration
% out.g_hist : the history value of the gradient norm of the differentiable part of the function
% out.f_hist : the history value of the objective function
% out.f_hist_best : the historical optimal value of the objective function at each iteration
% out.tt : runtime
% out.itr : number of iterations
% out.flag : record whether the algorithm has reached convergence

function [x, out] = l1_subgrad(x0, A, b, mu, opts)

% opts.maxit : maximum number of outer iterations
% opts.ftol : iteration termination condition for the function value
% opts.step_type : type of step-size
% opts.alpha0 : initial step-size
% opts.thres : threshold for determining whether a small amount is considered 0

if ~isfield(opts, 'maxit'); opts.maxit = 2000; end
if ~isfield(opts, 'thres'); opts.thres = 1e-4; end
if ~isfield(opts, 'step_type'); opts.step_type = 'fixed'; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 0.01; end
if ~isfield(opts, 'ftol'); opts.ftol = 0; end

x = x0;
out = struct();
out.f_hist = zeros(1, opts.maxit);
out.f_hist_best = zeros(1, opts.maxit);
out.g_hist = zeros(1, opts.maxit);
f_best = inf;

for k = 1:opts.maxit
    
    r = A * x - b;
    g = A' * r;
   
    out.g_hist(k) = norm(r, 2);
    
    f_now = 0.5 * norm(r, 2)^2 + mu * norm(x, 1);
    out.f_hist(k) = f_now;
    
    f_best = min(f_best, f_now);
    out.f_hist_best(k) = f_best;
    
    if k > 1 && abs(out.f_hist_best(k) - out.f_hist_best(k-1)) / abs(out.f_hist_best(1)) < opts.ftol
        break;
    end
    
    x(abs(x) < opts.thres) = 0;
    sub_g = g + mu * sign(x);
    
    alpha = set_step(k, opts);
    x = x - alpha * sub_g;
end

out.itr = k;
out.f_hist = out.f_hist(1:k);
out.f_hist_best = out.f_hist_best(1:k);
out.g_hist = out.g_hist(1:k);
end

% The function 'set_step' determines the step-size of the k-th step under different settings

function a = set_step(k, opts)
type = opts.step_type;
if strcmp(type, 'fixed')
    a = opts.alpha0;
elseif strcmp(type, 'diminishing')
    a = opts.alpha0 / sqrt(k);
elseif strcmp(type, 'diminishing2')
    a = opts.alpha0 / k;
else
    error('unsupported type.');
end
end