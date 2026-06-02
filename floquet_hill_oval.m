clear; clc; close all;

set(0,'defaultAxesFontSize',18);
set(0,'defaultLineLineWidth',1.4);

%% ============================================================
%  Floquet-Fourier-Hill code for oval spectral curves
%  Paper parameters:
%     sigma = +1, Delta = -2, k = 0.7
%  Figure 3 panels:
%     gamma = 0.2, 0.3225, 0.5
%% ============================================================

%% Parameters
sigma  = +1;
Delta  = -2;
kmod   = 0.7;

gamma_values = [0.2, 0.3225, 0.5];

% Numerical resolution
Nmodes = 60;    % truncation order for Hill matrix
Nmu    = 1200;  % dense Floquet sampling (increased) so closed oval loops are resolved

%% Base period L = 4 K(k)
Kk = ellipke(kmod^2);
L  = 4*Kk;

%% Plot limits
XLim = [-1 1];
YLim = [-3 3];

%% Precompute Fourier coefficients of psi^2 = k^2 sn^2(x,k)
Nx = 4096;
x  = linspace(0, L, Nx+1); x(end) = [];
[sn,~,~] = ellipj(x, kmod^2);
psi2 = (kmod*sn).^2;

% FFT on base period L:  psi2(x) = sum_n c_n exp(i 2*pi*n*x/L)
c_raw  = fftshift(fft(psi2)) / Nx;
nfull  = (-floor(Nx/2)) : (ceil(Nx/2)-1);

%% ============================================================
%  Figure
%% ============================================================
figure('Color','w','Position',[100 250 1280 380]);

for gi = 1:length(gamma_values)

    gamma = gamma_values(gi);
    % Paper relation uses tan(2*theta) = 2*gamma / (2*Delta + k^2 + 1).
    % Solve for theta with the half-angle form.
    theta = 0.5 * atan2(2*gamma, 2*Delta + kmod^2 + 1);

    % Floquet parameter mu sweeps one Brillouin zone [0, 2*pi/L)
    mu_vals = linspace(0, 2*pi/L, Nmu+1);
    mu_vals(end) = [];

    % Each mu gives 2*M eigenvalues (2x2 block system, M = 2*Nmodes+1)
    M_size   = 2 * (2*Nmodes+1);
    spec_all = zeros(Nmu * M_size, 1);
    idx0     = 0;

    for im = 1:length(mu_vals)
        mu  = mu_vals(im);
        lam = MI_Hill_oval_mu(sigma, Delta, kmod, gamma, theta, ...
                              L, Nmodes, mu, c_raw, nfull);
        n = length(lam);
        spec_all(idx0+1 : idx0+n) = lam;
        idx0 = idx0 + n;
    end

    spec_all = spec_all(1:idx0);

    % Keep only points inside the plot window (small margin for boundary eigenvalues)
    plot_margin = 0.05;
    mask = real(spec_all) >= XLim(1)-plot_margin & real(spec_all) <= XLim(2)+plot_margin & ...
           imag(spec_all) >= YLim(1)-plot_margin & imag(spec_all) <= YLim(2)+plot_margin;
    spec_plot = spec_all(mask);

    subplot(1,3,gi); hold on; box on;

    plot(real(spec_plot), imag(spec_plot), 'k.', 'MarkerSize', 2.5);
    h0 = plot([0 0], YLim, 'r--', 'LineWidth', 1.3);

    legend(h0, 'Re(\lambda) = 0', ...
        'Interpreter', 'tex', ...
        'Location',    'northeast', ...
        'FontSize',    11);

    xlabel('Re(\lambda)', 'Interpreter', 'tex', 'FontSize', 30);
    ylabel('Im(\lambda)', 'Interpreter', 'tex', 'FontSize', 30);

    xlim(XLim);
    ylim(YLim);
    xticks([-1 0 1]);
    yticks([-2 0 2]);

    set(gca, 'FontSize', 24, 'LineWidth', 1.0, 'TickDir', 'out');
    grid off;

    title(sprintf('\\gamma = %.4f', gamma), 'FontSize', 16);

    fprintf('gamma = %.4f,  max Re(lambda) = %.8f\n', ...
        gamma, max(real(spec_plot)));
end

%% ============================================================
%  Local function: Floquet-Hill spectrum at a single mu
%% ============================================================
function lambda = MI_Hill_oval_mu(sigma, Delta, kmod, gamma, theta, ...
                                   L, Nmodes, mu, c_raw, nfull)
    % Hill basis index vector
    j  = (-Nmodes : Nmodes).';
    M  = length(j);                    % = 2*Nmodes + 1

    % ---- Convolution matrix C for multiplication by psi^2 ----
    % C(a,b) = c_{j(a)-j(b)},  Fourier coefficient of psi^2.
    % Build a map from harmonic index n to position in nfull for O(M^2) lookup.
    n_min   = nfull(1);
    n_max   = nfull(end);
    n_range = n_max - n_min + 1;
    idx_map = zeros(1, n_range);   % idx_map(n - n_min + 1) = position in c_raw
    for ii = 1:length(nfull)
        idx_map(nfull(ii) - n_min + 1) = ii;
    end

    C = zeros(M, M);
    for a = 1:M
        for b = 1:M
            n = j(a) - j(b);
            if n >= n_min && n <= n_max
                pos = idx_map(n - n_min + 1);
                if pos > 0
                    C(a,b) = c_raw(pos);
                end
            end
        end
    end

    % ---- Floquet-shifted wavenumbers ----
    q  = mu + 2*pi*j / L;
    D2 = diag(-(q.^2));
    I  = eye(M);

    % ---- Matrix blocks (from paper) ----
    A11 =  -gamma*I + sigma*sin(2*theta)*C;
    A12 =  (-sigma*cos(2*theta) + 2*sigma)*C  - 0.5*D2;
    A21 =  -Delta*I + (-sigma*cos(2*theta) - 2*sigma)*C + 0.5*D2;
    A22 =  -gamma*I - sigma*sin(2*theta)*C;

    A = [A11, A12;
         A21, A22];

    lambda = eig(A);
end
