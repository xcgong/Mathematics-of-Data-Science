clear;
seed = 87916475;
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
opts.method = 'proximal_grad';
opts.verbose = 0;
opts.maxit = 4000;
opts.opts1 = struct();

opts.opts1.ls = 1;
opts.opts1.bb = 1;

opts.alpha0 = 1/L;
opts.ftol = 1e-12;
opts.gtol = 1e-10;

addpath('../LASSO_con')

[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = min(out.fvec);

% Proximal-gradient method with BB step-size and line search

opts = struct();
opts.method = 'proximal_grad';
opts.opts1 = struct();
opts.verbose = 0;
opts.maxit = 400;
opts.opts1.ls = 1;
opts.opts1.bb = 1;
opts.alpha0 = 1/L;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data1 = (out.fvec - f_star)/f_star;
k1 = min(length(data1),400);
data1 = data1(1:k1);

% Proximal-gradient method with fixed step-size

opts = struct();
opts.method = 'proximal_grad';
opts.opts1 = struct();
opts.verbose = 0;
opts.maxit = 400;
opts.opts1.ls = 0;
opts.opts1.bb = 0;
opts.alpha0 = 1/L;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data2 = (out.fvec - f_star)/f_star;
k2 = min(length(data2),400);
data2 = data2(1:k2);

% FISTA with BB step-size and line search

opts = struct();
opts.method = 'Nesterov';
opts.opts1 = struct();
opts.verbose = 0;
opts.maxit = 400;
opts.opts1.ls = 1;
opts.opts1.bb = 1;
opts.alpha0 = 1/L;
opts.ftol0 = 1;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data3 = (out.fvec - f_star)/f_star;
k3 = min(length(data3),400);
data3 = data3(1:k3);

% FISTA with fixed step-size

opts = struct();
opts.method = 'Nesterov';
opts.opts1 = struct();
opts.verbose = 0;
opts.maxit = 400;
opts.opts1.ls = 0;
opts.opts1.bb = 0;
opts.alpha0 = 1/L;
opts.ftol0 = 1;
[x, out] = LASSO_con(x0, A, b, mu, opts);
data4 = (out.fvec - f_star)/f_star;
k4 = min(length(data4),400);
data4 = data4(1:k4);

% Comparing the difference of convergence rate between the proximal-gradient method and FISTA

fig = figure;
semilogy(0:k1-1, data1, '-', 'Color',[0.99 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:k2-1, data2, ':','Color',[0.5 0.2 1], 'LineWidth',1.5);
hold on
semilogy(0:k3-1, data3, '-.','Color',[0.99 0.1 0.2], 'LineWidth',1.2);
hold on
semilogy(0:k4-1, data4, '--','Color',[0.2 0.1 0.99], 'LineWidth',1.5);
hold on
legend('Proximal-Gradient (BB)','Proximal-Gradient (Fixed)', 'FISTA (BB)','FISTA (Fixed)');
ylabel('$(f(x_k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('Iteration');
print(fig, '-depsc','fproxg.eps');