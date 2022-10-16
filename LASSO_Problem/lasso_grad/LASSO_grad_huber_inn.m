% Input information: model parameters A, b, mu, initial value of iteration x0, regularization parameter mu0 corresponding to the original problem, structure 'opts' containing algorithm parameters
% Output information: The iterative solution x (original LASSO problem) and a structure 'out' containing the iteration information

% out.fvec : the objective function value of the original LASSO problem at each iteration
% out.fval : the objective function value of the original LASSO problem at the termination of the iteration (corresponding to mu0 of the original problem)
% out.tt : runtime
% out.itr : number of iterations
% out.flag : record whether the algorithm has reached convergence

function [x, out] = LASSO_grad_huber_inn(x, A, b, mu, mu0, opts)

% opts.maxit : maximum number of outer iterations
% opts.ftol : iteration termination condition for the function value
% opts.gtol : iteration termination condition for gradient
% opts.alpha0 : initial step-size
% opts.sigma : Huber smoothing parameters
% opts.verbose : output each iteration information if not equal to 0, otherwise not output

if ~isfield(opts, 'maxit'); opts.maxit = 200; end
if ~isfield(opts, 'sigma'); opts.sigma = 0.1; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 0.01; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'verbose'); opts.verbose = 0; end
tt = tic;

r = A * x - b;
g = A' * r;

huber_g = sign(x);
idx = abs(x) < opts.sigma;
huber_g(idx) = x(idx) / opts.sigma;

g = g + mu * huber_g;
nrmG = norm(g,2);

f = .5*norm(r,2)^2 + mu*(sum(x(idx).^2/(2*opts.sigma)) + sum(abs(x(abs(x) >= opts.sigma)) - opts.sigma/2));

out = struct();
out.fvec = .5*norm(r,2)^2 + mu0*norm(x,1);

alpha = opts.alpha0;
eta = 0.2;
rhols = 1e-6;
gamma = 0.85;
Q = 1;
Cval = f;

for k = 1:opts.maxit
    
    fp = f;
    gp = g;
    xp = x;
   
    nls = 1;
    while 1
        
        x = xp - alpha*gp;
        r = A * x - b;
        g = A' * r;
        huber_g = sign(x);
        idx = abs(x) < opts.sigma;
        huber_g(idx) = x(idx) / opts.sigma;
        f = .5*norm(r,2)^2 + mu*(sum(x(idx).^2/(2*opts.sigma)) + sum(abs(x(abs(x) >= opts.sigma)) - opts.sigma/2));
        g = g + mu * huber_g;
        
%       Zhang-Hager line search
       
        if f <= Cval - alpha*rhols*nrmG^2 || nls >= 10
            break
        end
        alpha = eta*alpha;
        nls = nls+1;
    end
    
    nrmG = norm(g,2);
    forg = .5*norm(r,2)^2 + mu0*norm(x,1);
    out.fvec = [out.fvec, forg];
    
    if opts.verbose
        fprintf('%4d\t %.4e \t %.1e \t %.2e \t %2d \n',k, f, nrmG, alpha, nls);
    end

    if nrmG < opts.gtol || abs(fp - f) < opts.ftol
        break;
    end
   
%   Compute the BB step-size as the initial step size for the next iteration
   
    dx = x - xp;
    dg = g - gp;
    dxg = abs(dx'*dg);
    if dxg > 0
        if mod(k,2)==0
            alpha = dx'*dx/dxg;
        else
            alpha = dxg/(dg'*dg);
        end
        
        alpha = max(min(alpha, 1e12), 1e-12);
    end
   
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + f)/Q;
end

if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end

out.fval = f;
out.itr = k;
out.tt = toc(tt);
out.nrmG = nrmG;
end