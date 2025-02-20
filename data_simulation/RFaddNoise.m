function RFsyn_noised = RFAddNoise(RFsyn, tt, s2n, loco, hico)
    %
    % adding realistic noise to a reciever function
    %
    % s2n: SNR, 2.0
    % loco: lower cutoff frequency, 0.1
    % hico: higher cutoff frequency, 0.5

    % get nt, dt (and thus Fs) from tt
    nsample = length(tt);
    dt = diff(tt)(1);
    Fs = 1 ./ dt;
    % now we have sampling frequency Fs, and nsample

    nse = rnoise(RFsyn, s2n, 1:nsample, 0);
    nsefilt = bandpass(nse, Fs, loco, hico, 2);

    pn1 = sqrt(nse * nse');
    pn2 = sqrt(nsefilt' * nsefilt);
    scalar = pn1 / (pn2 * 1); % data variance preserced

    nsefilt = nsefilt .* scalar;

    RFsyn_noised = RFsyn + nsefilt'; % add to RF here
end
