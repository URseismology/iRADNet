function m = test_lip()
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

    x0 = rand(nt, nq);
    x0_freq = fft(real(x0), nfft, 1);

    [y1_freq, ~] = radon3d_forward(LLL, x0_freq, nt, ilow, ihigh);
    [~, x1] = radon3d_forward_adjoint(LLL, y1_freq, nt, ilow, ihigh);

    x1_normed = x1 ./ sqrt(sum(x1(:) .^ 2));
    x1_freq = fft(real(x1_normed), nfft, 1);

    [y2_freq, ~] = radon3d_forward(LLL, x1_freq, nt, ilow, ihigh);
    [~, x2] = radon3d_forward_adjoint(LLL, y2_freq, nt, ilow, ihigh);

    x2_normed = x2 ./ sqrt(sum(x2(:) .^ 2));
    x2_freq = fft(real(x2_normed), nfft, 1);

    [y3_freq, ~] = radon3d_forward(LLL, x2_freq, nt, ilow, ihigh);
    [~, x3] = radon3d_forward_adjoint(LLL, y3_freq, nt, ilow, ihigh);

    L = sum((x2_normed .* x3), "all") / sum((x2_normed .* x2_normed), "all");
    save('log/radon_test.mat', 'x0', 'x0_freq', 'y1_freq', 'x1', 'L', 'LLL');
    disp(L);
end
