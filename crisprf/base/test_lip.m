function m = lip_test()
    % test the lipschitz constant of the forward operator
    % also leave a trace of radon3d_forward intermiate outputs
    % for debugging and verifying correctness of pytorch implementation

    nt = 5000;
    nq = 200;
    np = 38;
    nfft = 16384;
    ilow = 2;
    ihigh = 8193;
    dt = 0.02;

    % def radon3d setup params
    N = 2;
    q = linspace(-1000, 1000, 200);
    rayP = linspace(0.098, 0.135, 38);

    % def radon3d matrix
    LLL = complex(zeros(nfft, np, nq));

    for ifreq = 1:nfft
        f = 2 .* pi * (ifreq - 1) / nfft / dt;
        LLL(ifreq, :, :) = exp(sqrt(-1) * f * (rayP .^ N)' * q);
    end

    x1 = rand(nt, nq);
    x1_freq = fft(real(x1), nfft, 1);

    [y2_freq, ~] = radon3d_forward(LLL, x1_freq, nt, ilow, ihigh);
    [~, x2] = radon3d_forward_adjoint(LLL, y2_freq, nt, ilow, ihigh);

    x2_norm = sqrt(sum(x2(:) .^ 2));
    x2_normed = x2 ./ x2_norm;
    x2_freq = fft(real(x2_normed), nfft, 1);

    [y3_freq, ~] = radon3d_forward(LLL, x2_freq, nt, ilow, ihigh);
    [~, x3] = radon3d_forward_adjoint(LLL, y3_freq, nt, ilow, ihigh);

    x3_norm = sqrt(sum(x3(:) .^ 2));
    x3_normed = x3 ./ x3_norm;
    x3_freq = fft(real(x3_normed), nfft, 1);

    [y4_freq, ~] = radon3d_forward(LLL, x3_freq, nt, ilow, ihigh);
    [~, x4] = radon3d_forward_adjoint(LLL, y4_freq, nt, ilow, ihigh);

    L = sum((x3_normed .* x4), "all") / sum((x3_normed .* x3_normed), "all");
    save('log/radon_test.mat', 'x1', 'x1_freq', 'y2_freq', 'x2', 'x2_norm', 'x2_normed', 'x2_freq', 'y3_freq', 'x3', 'x3_norm', 'x3_normed', 'x3_freq', 'y4_freq', 'x4', 'L', 'LLL');
    disp(L);
end
