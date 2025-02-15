function m = sparse_inverse_radon_fista(d, dt, rayP, q, ...
        flow, fhigh, reg_param, reg_param2, maxiter)
    %%SPARSE_INVERSE_RADON_FISTA Calculates SRTFISTA-based reconstruction on
    %%real data
    %
    % Inputs:
    %       d:         seismic traces
    %       dt:        sampling in sec
    %       rayP:      ray parameters
    %       q:         q (curvature) range as a vector
    %       flow:      freq.  where the inversion starts in Hz (> 0 Hz)
    %       fhigh:     freq.  where the inversion ends in Hz (< Nyquist)
    %       reg_param: regularization parameter (denoted by lambda in the notes)
    %       reg_param2:regularization parameter of the L2 regularized solution
    %            used as the starting point of the algorithm
    %       maxiter:   maximum number of iterations
    %
    % Outputs:
    %       m:         reconstructed Radon image
    %

    %% Initialization
    N = 2; % always use parabolic

    [nt, np] = size(d);
    nq = max(size(q));

    nfft = 2 * (2 ^ nextpow2(nt));

    % initialize model matrices in frequncy domain
    D = fft(d, nfft, 1);
    M0 = complex(zeros(nfft, nq));

    % frequency set up for kernel initialization
    ilow = floor(flow * dt * nfft) + 1;
    if ilow < 2; ilow = 2; end
    ihigh = floor(fhigh * dt * nfft) + 1;
    if ihigh > floor(nfft / 2) + 1; ihigh = floor(nfft / 2) + 1; end
    ilow = max(ilow, 2);

    % initialize kernel matrix (3D) in freq. domain
    LLL = complex(zeros(nfft, np, nq));

    for ifreq = 1:nfft
        f = 2 .* pi * (ifreq - 1) / nfft / dt;
        LLL(ifreq, :, :) = exp(sqrt(-1) * f * (rayP .^ N)' * q);
    end

    %% Perform projection on the data
    % Calculate the starting point of the algorithm with L2 regularization.
    Q = eye(nq);

    for ifreq = ilow:ihigh

        L = squeeze(LLL(ifreq, :, :));
        y = D(ifreq, :)';

        xa = L' * y;
        Ab = L' * L;

        A = Ab + reg_param2 * Q;

        x = A \ xa;

        M0(ifreq, :) = x';
        M0(nfft + 2 - ifreq, :) = conj(x)';

    end

    M0(nfft / 2 + 1, :) = complex(zeros(1, nq));
    m0 = real(ifft(M0, [], 1));
    m0 = m0(1:nt, :);

    %% Calculate the step size
    % To calculate the step size, we estimate the maximum eigenvalue of A^H*A
    % using power iterations. Here, L is the resulting estimate. Here, we only
    % perform 2 iterations, but it can be increased to improve the accuracy of
    % the estimate.
    %
    % https://en.wikipedia.org/wiki/Power_iteration

    b_k = rand(nt, nq);
    B_k = fft(real(b_k), nfft, 1);

    for i = 1:2
        [B_k1, ~] = radon3d_forward(LLL, B_k, nt, ilow, ihigh);
        [~, b_k1] = radon3d_forward_adjoint(LLL, B_k1, nt, ilow, ihigh);
        b_k1_norm = sqrt(sum(b_k1(:) .^ 2));
        b_k = b_k1 ./ b_k1_norm;
        B_k = fft(real(b_k), nfft, 1);
    end

    [B_k_temp, ~] = radon3d_forward(LLL, B_k, nt, ilow, ihigh);
    [~, b_k_temp] = radon3d_forward_adjoint(LLL, B_k_temp, nt, ilow, ihigh);
    L = sum((b_k .* b_k_temp), "all") / sum((b_k .* b_k), "all");

    %% FISTA
    % initialize the algorithm.
    m = m0;
    s = m;
    q_t = 1;
    step_size = 1 / L * 0.9;

    for kstep = 1:maxiter
        % z-update
        [temp, ~] = radon3d_forward(LLL, fft(real(s), nfft, 1), nt, ilow, ihigh);
        [~, temp] = radon3d_forward_adjoint(LLL, temp - D, nt, ilow, ihigh);
        z = s - step_size * temp;
        % m-update
        m_prev = m;
        m = wthresh(z, 's', step_size * reg_param);
        % q-update
        q_new = 0.5 * (1 + sqrt(1 + 4 * (q_t ^ 2)));
        % s-update
        s = m + (q_t - 1) / q_new * (m - m_prev);
        q_t = q_new;

        pause(.0002)
    end
