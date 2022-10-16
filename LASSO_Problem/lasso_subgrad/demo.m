clear;
seed = 87016475;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);

m = 512;
n = 1024;
A = randn(m, n);
u = sprandn(n, 1, 0.1);
b = A * u;

mu = 1;
x0 = u + 1e-1 * randn(n, 1);

% Conduct experiments under stricter convergence conditions, and use the function value obtained at the termination of the iteration as the reference for the real optimal value f*

opts_g = struct();
opts_g.maxit = 50000;
opts_g.alpha0 = 1e-6;
opts_g.step_type = 'fixed';
[x_g, out_g] = l1_subgrad(u, A, b, mu, opts_g);
f_star = out_g.f_hist_best(end);

% Subgradient method with fixed step-size

opts = struct();
opts.maxit = 3000;
opts.alpha0 = 0.0005;
opts.step_type = 'fixed';
[x, out] = l1_subgrad(x0, A, b, mu, opts);
data1 = (out.f_hist - f_star) / f_star;
data1_best = (out.f_hist_best - f_star) / f_star;

opts.alpha0 = 0.0002;
[x, out] = l1_subgrad(x0, A, b, mu, opts);
data2 = (out.f_hist - f_star) / f_star;
data2_best = (out.f_hist_best - f_star) / f_star;

opts.alpha0 = 0.0001;
[x, out] = l1_subgrad(x0, A, b, mu, opts);
data3 = (out.f_hist - f_star) / f_star;
data3_best = (out.f_hist_best - f_star) / f_star;

% Subgradient method with diminishing step-sizes

opts.step_type = 'diminishing';
opts.alpha0 = 0.002;
[x, out] = l1_subgrad(x0, A, b, mu, opts);
data4 = (out.f_hist - f_star) / f_star;
data4_best = (out.f_hist_best - f_star) / f_star;

% The relative error of the objective function value obtained at each iteration

fig = figure;
semilogy(0:length(data1)-1, data1, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:length(data2)-1, data2, '--','Color',[0.99 0.1 0.2], 'LineWidth',1.2);
hold on
semilogy(0:length(data3)-1, data3, '-.','Color',[0.99 0.1 0.99], 'LineWidth',1.5);
hold on
semilogy(0:length(data4)-1, data4, ':','Color',[0.5 0.2 0.1], 'LineWidth',1.8);
hold on
legend('0.0005', '0.0002', '0.0001', '$0.002/{\sqrt{k}}$','interpreter', 'latex');
ylabel('$(f(x_k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('Iteration');
print(fig, '-depsc','f.eps');

% The relative error of the historical optimal value of the objective function obtained at each iteration

fig = figure;
semilogy(0:length(data1_best)-1, data1_best, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:length(data2_best)-1, data2_best, '--','Color',[0.99 0.1 0.2], 'LineWidth',1.2);
hold on
semilogy(0:length(data3_best)-1, data3_best, '-.','Color',[0.99 0.1 0.99], 'LineWidth',1.5);
hold on
semilogy(0:length(data4_best)-1, data4_best, ':','Color',[0.5 0.2 0.1], 'LineWidth',1.8);
hold on
legend('0.0005', '0.0002', '0.0001', '$0.002/{\sqrt{k}}$','interpreter', 'latex');
ylabel('$(\hat{f}(x_k) - f^*)/f^*$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('Iteration');
print(fig, '-depsc','f_best.eps');