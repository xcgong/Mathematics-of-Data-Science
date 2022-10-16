% Input information: A, b, mu0, the initial value of the iteration x0 and the structure 'opts' providing each parameter.
% Output information: The iterative solution x and the structure 'out'.

% out.fvec : function value for each iteration
% out.itr_inn : total inner iterations
% out.fval : the objective function value when the iteration terminates
% out.tt : runtime
% out.itr : number of outer iterations

function [x, out] = LASSO_con(x0, A, b, mu0, opts)

% opts.maxit : maximum number of outer iterations
% opts.maxit_inn : maximum number of inner iterations
% opts.ftol : iteration termination condition for the function value
% opts.gtol : iteration termination condition for gradient
% opts.factor : decay rate of the regularization parameter
% opts.verbose : output each iteration information if not equal to 0, otherwise not output
% opt.mu1 : initial regularization parameters (with continuation strategy, starting with larger regularization parameters)
% opts.alpha0 : initial step-size
% opts.ftol_init_ratio : magnification of the initial iteration termination condition 'opts.ftol'
% opts.gtol_init_ratio : magnification of the initial iteration termination condition 'opts.gtol'
% opts.etaf : reduction of the iteration termination condition 'opts.ftol' for the outer loop at each step
% opts.etag : reduction of the iteration termination condition 'opts.gtol' for the outer loop at each step
% opts.opts1 : structure, used to provide other specific parameters to the inner algorithm

if ~isfield(opts, 'maxit'); opts.maxit = 30; end
if ~isfield(opts, 'maxit_inn'); opts.maxit_inn = 200; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'factor'); opts.factor = 0.1; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end
if ~isfield(opts, 'mu1'); opts.mu1 = 100; end
if ~isfield(opts, 'gtol_init_ratio'); opts.gtol_init_ratio = 1/opts.gtol; end
if ~isfield(opts, 'ftol_init_ratio'); opts.ftol_init_ratio = 1e5; end
if ~isfield(opts, 'opts1'); opts.opts1 = struct(); end
if ~isfield(opts, 'etaf'); opts.etaf = 1e-1; end
if ~isfield(opts, 'etag'); opts.etag = 1e-1; end
L = eigs(A'*A,1);
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1/L; end

% The algorithm for solving the subproblem corresponding to the regularization parameter mu_t is selected by 'opts.method'

if ~isfield(opts, 'method'); error('Need opts.method'); end
algf = eval(sprintf('@LASSO_%s_inn',opts.method));

addpath(genpath(pwd));

out = struct();
out.fvec = [];
k = 0;
x = x0;
mu_t = opts.mu1;
tt = tic;

f = Func(A, b, mu_t, x);

opts1 = opts.opts1;
opts1.ftol = opts.ftol*opts.ftol_init_ratio;
opts1.gtol = opts.gtol*opts.gtol_init_ratio;
out.itr_inn = 0;

while k < opts.maxit
    
%     The inner loop parameter settings are recorded in the structure opts1
% 
%     opts1.itr: maximum number of iterations, given by opts.maxit_inn
%     opts1.ftol: iteration termination condition for function value
%     opts1.gtol: iteration termination condition for gradient
%     opts1.alpha0: initial step size
%     opts1.verbose: true when 'ops.verbose' is greater than 1, then detailed output of inner iteration information
    
    opts1.maxit = opts.maxit_inn;
    opts1.gtol = max(opts1.gtol * opts.etag, opts.gtol);
    opts1.ftol = max(opts1.ftol * opts.etaf, opts.ftol);
    opts1.verbose = opts.verbose > 1;
    opts1.alpha0 = opts.alpha0;
    
    if strcmp(opts.method, 'grad_huber'); opts1.sigma = 1e-3*mu_t; end
    
    fp = f;
    [x, out1] = algf(x, A, b, mu_t, mu0, opts1);
    f = out1.fvec(end);
    out.fvec = [out.fvec, out1.fvec];
    k = k + 1;
    
    nrmG = norm(x - prox(x - A'*(A*x - b),mu0),2);
    
    if opts.verbose
        fprintf('itr: %d\tmu_t: %e\titr_inn: %d\tfval: %e\tnrmG: %.1e\n', k, mu_t, out1.itr, f, nrmG);
    end
   
    if ~out1.flag
        mu_t = max(mu_t * opts.factor, mu0);
    end

    if mu_t == mu0 && (nrmG < opts.gtol || abs(f-fp) < opts.ftol)
        break;
    end
    
    out.itr_inn = out.itr_inn + out1.itr;
end

out.fval = f;
out.tt = toc(tt);
out.itr = k;

%   Objective function for the original LASSO problem

    function f = Func(A, b, mu0, x)
        w = A * x - b;
        f = 0.5 * (w' * w) + mu0 * norm(x, 1);
    end

%   Proximal operator for function mu||x||_1

    function y = prox(x, mu)
        y = max(abs(x) - mu, 0);
        y = sign(x) .* y;
    end
end