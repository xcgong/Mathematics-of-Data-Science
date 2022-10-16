clear;
seed = 87016475;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);

m = 512;
n = 1024;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;
x0 = randn(n, 1);
mu = 1e-3;

L = eigs(A'* A, 1);

% Conduct experiments under stricter convergence conditions, and use the function value obtained at the termination of the iteration as the reference for the real optimal value f*

opts = struct();
opts.maxit = 5000;
opts.maxit_inn = 500;
opts.opts1 = struct();
opts.method = 'subgrad';
opts.opts1.step_type = 'diminishing';
opts.verbose = 0;
opts.alpha0 = 1/L;
opts.ftol = 1e-12;
opts.ftol0 = 1e4;
opts.etag = 1;

addpath('../LASSO_con')

[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = out.fvec(end);

% Subgradient method with continuation strategy

opts.maxit = 3000;
opts.maxit_inn = 200;
opts.opts1.step_type = 'diminishing';
opts.verbose = 0;
opts.ftol = 1e-8;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data1 = (out.fvec - f_star) / f_star;
k1 = length(data1);

% Modify mu to 1e-2 to repeat the experiment

mu = 1e-2;
opts.maxit = 5000;
opts.maxit_inn = 500;
opts.opts1.step_type = 'fixed';
opts.ftol = 1e-10;
[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = out.fvec(end);

opts.maxit = 3000;
opts.maxit_inn = 200;
opts.ftol = 1e-8;
opts.opts1.step_type = 'fixed';
[x, out] = LASSO_con(x0, A, b, mu, opts);
data2 = (out.fvec - f_star) / f_star;
k2 = length(data2);

% Observe how the objective function value changes with the number of iterations

fig = figure;
semilogy(1:k1, max(data1,0), '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(1:k2, max(data2,0), '-.','Color',[0.99 0.1 0.2], 'LineWidth',1.5);
legend('$\mu = 10^{-3}$', '$\mu = 10^{-2}$','interpreter', 'latex');
ylabel('$(f(x_k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('Iteration');
print(fig, '-depsc','subgrad.eps');