% Input information: model parameters A, b, mu, initial value of iteration x0, regularization parameter mu0 corresponding to the original problem, structure 'opts' containing algorithm parameters
% Output information: The iterative solution x (original LASSO problem) and a structure 'out' containing the iteration information

% out.fvec : the objective function value of the original LASSO problem at each iteration
% out.g_hist : the history value of the gradient norm of the differentiable part of the function
% out.f_hist : the current objective function value at each iteration (corresponding to the current mu_t)
% out.f_hist_best : the historical optimal value of the current objective function at each iteration (corresponding to the current mu_t)
% out.tt : runtime
% out.itr : number of iterations
% out.flag : record whether the algorithm has reached convergence

function [x, out] = LASSO_subgrad_inn(x, A, b, mu, mu0, opts)

% opts.maxit : maximum number of outer iterations
% opts.ftol : iteration termination condition for the function value
% opts.step_type : type of step-size
% opts.alpha0 : initial step-size
% opts.thres : threshold for determining whether a small amount is considered 0

if ~isfield(opts, 'maxit'); opts.maxit = 500; end
if ~isfield(opts, 'thres'); opts.thres = 1e-4; end
if ~isfield(opts, 'step_type'); opts.step_type = 'fixed'; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 0.01; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-12; end

if mu > mu0
    opts.step_type = 'fixed';
else
    opts.step_type = opts.step_type;
end

tic;
out = struct();
out.fvec = [];
r = A * x - b;
gx = A' * r;
sub_g = gx + mu * sign(x);
f_best = inf;

for k = 1:opts.maxit
    alpha = set_step(k, opts);
    x = x - alpha * sub_g;
    r = A * x - b;
    g = A' * r;
   
    x(abs(x) < opts.thres) = 0;
    sub_g = g + mu * sign(x);
    
    out.grad_hist(k) = norm(r, 2);
    tmp = .5*norm(r,2)^2;
    nrmx1 = norm(x,1);
    f = tmp + mu * nrmx1;
    
    out.f_hist(k) = f;
    f_best = min(f_best, f);
    out.f_hist_best(k) = f_best;
    out.fvec = [out.fvec, tmp + mu0*nrmx1];
    
    if opts.verbose
        fprintf('itr: %4d \t f: %.4e \t step: %.1e\n',k, f, alpha);
    end
    
    FDiff = abs(out.f_hist(k) - out.f_hist(max(k-1,1))) / abs(out.f_hist_best(1));
    BFDiff = abs(out.f_hist_best(max(k - 8,1)) - min(out.f_hist_best(max(k-7,1):k)));
    if (k > 1 && FDiff < opts.ftol) || (k > 8 && BFDiff < opts.ftol)
        break;
    end
end

if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end

out.itr = k;
out.tt = toc;
end

% The function 'set_step' determines the step-size of the k-th step under different settings

function a = set_step(k, opts)
type = opts.step_type;
if strcmp(type, 'fixed')
    a = opts.alpha0;
elseif strcmp(type, 'diminishing')
    a = opts.alpha0 / sqrt(max(k,100)-99);
elseif strcmp(type, 'diminishing2')
    a = opts.alpha0 / (max(k,100)-99);
else
    error('unsupported type.');
end
end