function [d] = forward_radon_freq_umd_Sp(m, dt, p, q, q1, q0, flow, fhigh)
    %FORWARD_RADON_FREQ: Forward linear and parabolic Radon transform.
    %                    Freq. domain algorithm
    %
    %  [d] = forward_radon_freq(m,dt,h,p,N,flow,fhigh);
    %
    %  IN   m:     the Radon panel, a matrix m(nt,np)
    %       dt:    sampling in sec
    %       p(nh): ray param
    %       q(np): ray parameter  to retrieve if N=1
    %              curvature of the parabola if N=2
    %       q1:    1st-order term

    %       flow:  min freq. in Hz
    %       fhigh: max freq. in Hz
    %
    %  OUT  d:     data
    %
    %  Reference: Hampson, D., 1986, Inverse velocity stacking for multiple elimination,
    %             Journal of the CSEG, vol 22, no 1., 44-55.
    %
    %
    %  Copyright (C) 2008, Signal Analysis and Imaging Group
    %  For more information: http://www-geo.phys.ualberta.ca/saig/SeismicLab
    %  Author: M.D.Sacchi
    %
    %  This program is free software: you can redistribute it and/or modify
    %  it under the terms of the GNU General Public License as published
    %  by the Free Software Foundation, either version 3 of the License, or
    %  any later version.
    %
    %  This program is distributed in the hope that it will be useful,
    %  but WITHOUT ANY WARRANTY; without even the implied warranty of
    %  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    %  GNU General Public License for more details: http://www.gnu.org/licenses/
    %

    [nt, ~] = size(m);
    np = length(p);

    nfft = 2 * (2 ^ nextpow2(nt));

    M = fft(m, nfft, 1);
    D = zeros(nfft, np);
    i = sqrt(-1);

    ilow = floor(flow * dt * nfft) + 1;
    if ilow < 1; ilow = 1; end
    ihigh = floor(fhigh * dt * nfft) + 1;
    if ihigh > floor(nfft / 2) + 1; ihigh = floor(nfft / 2) + 1; end

    for ifreq = ilow:ihigh
        f = 2 .* pi * (ifreq - 1) / nfft / dt;
        L1 = exp(i * f * (p' .^ 2 * q));
        L2 = exp(i * f * (p' * repmat(q1, size(q)))); % for linear term - scalar
        % L3 = exp(i*f*repmat(q0,size(q))); % for constant term - scalar
        x = M(ifreq, :)';
        y = (L1 .* L2) * x;
        D(ifreq, :) = y';
        D(nfft + 2 - ifreq, :) = conj(y)';
    end

    D(nfft / 2 + 1, :) = zeros(1, np);
    d = real(ifft(D, [], 1));
    d = d(1:nt, :);

    return;
