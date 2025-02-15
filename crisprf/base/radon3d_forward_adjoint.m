function [M, m] = radon3d_forward_adjoint(LLL, DD, nt, ilow, ihigh);

    [nfft, ~, nq] = size(LLL);
    M = complex(zeros(nfft, nq));

    for ifreq = ilow:ihigh
        % --- transform here
        L = squeeze(LLL(ifreq, :, :));
        y = DD(ifreq, :)';
        x = L' * y;
        % ---- done

        M(ifreq, :) = x';
        M(nfft + 2 - ifreq, :) = conj(x)';
    end

    M(nfft / 2 + 1, :) = zeros(1, nq);
    m = real(ifft(M, [], 1));
    m = m(1:nt, :);

    return;
