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

L = eigs(A'*A, 1);

% Conduct experiments under stricter convergence conditions, and use the function value obtained at the termination of the iteration as the reference for the real optimal value f*

opts = struct();
opts.method = 'grad_huber';
opts.verbose = 0;
opts.maxit = 4000;
opts.ftol = 1e-8;
opts.alpha0 = 1 / L;

addpath('../LASSO_con')

[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = min(out.fvec);

% Huber smoothing gradient method with continuation strategy, along with BB step-size and line search

opts.verbose = 0;
opts.maxit = 400;
if opts.verbose
    fprintf('mu=1e-3\n');
end
[x, out] = LASSO_con(x0, A, b, mu, opts);
data1 = (out.fvec - f_star)/f_star;
k1 = min(length(data1),400);
data1 = data1(1:k1);

% Modify mu to 1e-2 to repeat the experiment

mu = 1e-2;
opts = struct();
opts.method = 'grad_huber';
opts.verbose = 0;
opts.maxit = 4000;
opts.ftol = 1e-8;
opts.alpha0 = 1 / L;
[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = min(out.fvec);

opts.verbose = 0;
opts.maxit = 400;
if opts.verbose
    fprintf('\nmu=1e-2\n');
end
[x, out] = LASSO_con(x0, A, b, mu, opts);
data2 = (out.fvec - f_star)/f_star;
k2 = min(length(data2),400);
data2 = data2(1:k2);

% Observe how the objective function value changes with the number of iterations

fig = figure;
semilogy(0:k1-1, data1, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:k2-1, data2, '-.','Color',[0.99 0.1 0.2], 'LineWidth',1.5);
legend('$\mu = 10^{-3}$', '$\mu = 10^{-2}$','interpreter', 'latex');
ylabel('$(f(x_k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('Iteration');
print(fig, '-depsc','grad.eps');

% The magnitude of each component of the solution obtained by the gradient method

fig = figure;
subplot(2, 1, 1);
plot(u, 'Color',[0.2 0.1 0.99], 'Marker', 'x', 'LineStyle', 'none');
xlim([1, 1024]);
title('Exact Solution');

subplot(2, 1, 2);
plot(x, 'Color',[0.2 0.1 0.99], 'Marker', 'x', 'LineStyle', 'none');
xlim([1, 1024]);
title('Huber Smoothing Gradient Method Solution');
print(fig, '-depsc','solu-smoothgrad.eps');