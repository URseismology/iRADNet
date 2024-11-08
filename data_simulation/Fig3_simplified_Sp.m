tshift = 10;

%% define radon synthesis panel
rayP = (0.098:0.001:0.135); % #  of events

% intercept time resolution --
dt = 0.02; %time sample size -- data receiver functions
nt = 5000; % number of taus
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
Min = zeros(nq, nt); %initialize Min (radon domain)
Min_2 = zeros(nq, nt); %Sharper q image

for ilyr = 1:nlayers

    for jphs = 1:nPhases

        gauswvlt = abs(m(ilyr, jphs)) .* stat(gwr, gtwr, tau(ilyr, jphs)); %shift pulses in time. A sample at time t on input will be at t+delt on output

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

%% Plot

twin = [0 30];
qwin = [-800 800];
figure;
clf;
% RadonPlot(taus, qs, Min_2', twin, qwin, [], []);

[t1, t2, t3] = travelTimesAppx([Vs 4.5], [Vp 8.1], [H 0], rayP, 1, 2);

twin = [-10 30];
% RFWigglePlot(tx, taus - tshift, rayP, rayP, [t1(end:-1:1) t2(end:-1:1) t3(end:-1:1)], twin, 1, 1, 0);

%% Save RF
save('Sp_RF_syn.mat', 'tx', 'taus', 'rayP', 'tshift', 'Min_2');
