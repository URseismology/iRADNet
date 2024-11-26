function data_sim(tshift, nt, dt, save_path)
    % save one synthetic sample to `save_path`
    % Params:
    % tshift: int, time shift
    % nt: int, number of taus
    % dt: float, time sample size

    % tshift = 10;
    % nt = 5000; % number of taus
    % dt = 0.02; % time sample size -- data receiver functions

    %% define radon synthesis panel
    rayP = (0.098:0.001:0.135);

    % intercept time resolution --
    taus = (0:1:nt - 1) * dt; %generate taus scaled by the sample time
    tlen = ceil(taus(end));

    % travel time curvature resolution
    absq = 1000;
    qmin = -absq;
    qmax = absq; % Radon function parameters
    nq = 200;
    dq = (qmax - qmin) / (nq - 1); %value increament of qs
    qs = qmin + dq * (0:1:nq - 1); % generate all q values from qmin to qmax
    nlayers = 1; % each layer generates 3 phases
    nPhases = 3; % each phase: Ps, and 2 crustal multiples

    %set velocity model
    % TODO: figure out what values to pass from outside
    Vp = [6.3 8.1];
    Vs = [3.6 4.5];
    H = [35 0];

    [tau, q, q1, q0] = get_q_t_Sp(H, Vp, Vs);
    tau = tau + tshift;

    m = [1 0.5 -0.5];
    m = m ./ max(abs(m), [], 'all');

    %-- frequency and wavelets
    fmin = 0;
    fmax = 1 / (2 * dt);
    fbw = 5; % bandwidth of 1 Hz

    %--Generating Gaussian Wave
    [gwr, gtwr] = gauswvlet(dt, fbw, 2 * tlen);

    % --Creating Radon doamin
    % Min = zeros(nq, nt); %initialize Min (radon domain)
    Min_2 = zeros(nq, nt); %Sharper q image

    for ilyr = 1:nlayers

        for jphs = 1:nPhases

            %shift pulses in time. A sample at time t on input will be at t+delt on output
            gauswvlt = abs(m(ilyr, jphs)) .* stat(gwr, gtwr, tau(ilyr, jphs));

            if m(ilyr, jphs) < 0
                gauswvlt = phsrot(gauswvlt, 180); % rotate this ricker wavelet to negative phase
            end

            indq = find(qs > q(ilyr, jphs), 1);

            % [q(ilyr, jphs), tau(ilyr, jphs), m(ilyr, jphs)]

            %     Min(indq-1, :) = gauswvlt(end-nt+1:end);
            %     Min(indq, :) = gauswvlt(end-nt+1:end);
            %     Min(indq+1, :) = gauswvlt(end-nt+1:end);

            Min_2(indq, :) = gauswvlt(end - nt + 1:end);

        end

    end

    mm = max(abs(Min_2), [], 'all'); Min_2 = Min_2 ./ mm;

    [tx] = forward_radon_freq_umd_Sp(Min_2', dt, rayP, qs, q1, q0, fmin, fmax);
    tx = tx';
    [dx, dy] = size(tx);

    save(save_path, 'tx', 'taus', 'rayP', 'tshift', 'Min_2');
