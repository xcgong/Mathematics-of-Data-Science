% The function completes the optimization of the inner iteration under the continuation strategy for LASSO problem

% Input information: model parameters A, b, mu, initial value of iteration x0, regularization parameter mu0 corresponding to the original problem, structure 'opts' containing algorithm parameters
% Output information: The iterative solution x (original LASSO problem) and a structure 'out' containing the iteration information

% out.fvec : the objective function value of the original LASSO problem at each iteration
% out.fval : the objective function value of the original LASSO problem at the termination of the iteration (corresponding to mu0 of the original problem)
% out.nrmG : the gradient norm at the termination of the iteration
% out.tt : runtime
% out.itr : number of iterations
% out.flag : record whether the algorithm has reached convergence

function [x, out] = LASSO_proximal_grad_inn(x0, A, b, mu, mu0, opts)

% opts.maxit : maximum number of outer iterations
% opts.ftol : iteration termination condition for the function value
% opts.gtol : iteration termination condition for gradient
% opts.alpha0 : initial step-size
% opts.verbose : output each iteration information if not equal to 0, otherwise not output
% opts.ls : mark whether to use line search
% opts.bb : mark whether to take BB step-size

if ~isfield(opts, 'maxit'); opts.maxit = 10000; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-12; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1e-3; end
if ~isfield(opts, 'ls'); opts.ls = 1; end
if ~isfield(opts, 'bb'); opts.bb = 0; end

out = struct();
k = 0;
tt = tic;
x = x0;
t = opts.alpha0;

fp = inf;
r = A*x0 - b;
g = A'*r;
tmp = .5*norm(r,2)^2;
tmpf = tmp + mu*norm(x,1);
f =  tmp + mu0*norm(x,1);
nrmG = norm(x - prox(x - g,mu),2);
out.fvec = f;

Cval = tmpf; Q = 1; gamma = 0.85; rhols = 1e-6;

while k < opts.maxit && nrmG > opts.gtol && abs(f - fp) > opts.ftol
    
    gp = g;
    fp = f;
    xp = x;
    
    x = prox(xp - t * g, t * mu);
    
%   Zhang-Hager line search
    
    if opts.ls
        nls = 0;
        while 1
            tmp = 0.5 * norm(A*x - b, 2)^2;
            tmpf = tmp + mu*norm(x,1);
            if tmpf <= Cval - rhols*0.5*norm(x-xp,2)^2/t || nls == 5
                break;
            end
            
            t = 0.2*t; nls = nls + 1;
            x = prox(xp - t * g, t * mu);
        end
        
        f = tmp + mu0*norm(x,1);
       
    else
        f = 0.5 * norm(A*x - b, 2)^2 + mu0*norm(x,1);
    end
   
    nrmG = norm(x - xp,2)/t;
    r = A * x - b;
    g = A' * r;
    
%   Compute the BB step-size as the initial step size for the next iteration
    
    if opts.bb && opts.ls
        dx = x - xp;
        dg = g - gp;
        dxg = abs(dx'*dg);
        
        if dxg > 0
            if mod(k,2) == 0
                t = norm(dx,2)^2/dxg;
            else
                t = dxg/norm(dg,2)^2;
            end
        end
        
        t = min(max(t,opts.alpha0),1e12);
        Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + tmpf)/Q;
        
    else
        t = opts.alpha0;
    end
    
    k = k + 1;
    out.fvec = [out.fvec, f];
    if opts.verbose
        fprintf('itr: %d\tt: %e\tfval: %e\tnrmG: %e\n', k, t, f, nrmG);
    end
  
    if k > 8 && min(out.fvec(k-7:k)) - out.fvec(k-8) > opts.ftol
        break;
    end
end

if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end

out.fvec = out.fvec(1:k);
out.fval = f;
out.itr = k;
out.tt = toc(tt);
out.nrmG = nrmG;
end

% Proximal operator for function mu||x||_1

function y = prox(x, mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end