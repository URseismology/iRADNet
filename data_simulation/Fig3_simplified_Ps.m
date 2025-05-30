%%
tshift = 0;

% define radon synthesis panel
clc
rayP = (0.098:0.001:0.135); % #  of events

% intercept time resolution --
dt = 0.02;           %time sample size -- data receiver functions
nt = 5000;        % number of taus
taus = (0:1:nt-1) * dt; %generate taus scaled by the sample time
tlen = ceil(taus(end));

% travel time curvature resolution
absq = 1000;
qmin = -absq;
qmax = absq;                % Radon function parameters
nq = 200;
dq = (qmax-qmin)/(nq-1);  %value increament of qs
qs = qmin+dq*(0:1:nq-1);   % generate all q values from qmin to qmax


nPhases = 3;  % each phase: Ps, and 2 crustal multiples

%set velocity model

% Number of samples
n_samples = 100;


%% set velocity model
if 0
    nlayers = 3;  % each layer generates 3 phases
    H  = [18 14 82 0]; %35 used to be 52
    Vp = [6.22 6.54 8.20 8.20];
    Vs = [3.57 3.76 4.60 4.14];
    rho = [2.600 2.600 2.900 3.300];
else
    nlayers = 1;  % each layer generates 3 phases
    Vp = [6.3 8.1];
    Vs = [3.6 4.5];
    H  = [35 0];


    % Generate random samples within the specified ranges
    Vp_samples = generate_samples(Vp, n_samples, 1.0);
    Vs_samples = generate_samples(Vs, n_samples, 0.8);
    
    range2 = H;
    H_samples = range2(1) + (range2(2) - range2(1)) * rand(n_samples, 2); 
    H_samples(:,2) = 0;


end

%%
for isample = 1%: n_samples

    Vpi = Vp_samples(isample,:);
    Vsi = Vs_samples(isample,:);
    Hi = H_samples(isample,:);

    [tau, q, q1, q0] = get_q_t_Sp(Hi, Vsi, Vpi);
    %[q, tau] = get_q_t(H, Vp, Vs);

    tau = tau + tshift;

    %m=[1 0.5 -0.5];
    m=[0.31 0.50 -0.34;1 0.58 -0.19;-0.58 0.122 -0.122]'; % radon amplitude for each phase
    m = m ./max(abs(m), [], 'all') ;

    %-- frequency and wavelets
    fmin = 0;
    fmax = 1/(2*dt);
    fbw = 5 ;  % bandwidth of 1 Hz

    %--Generating Gaussian Wave
    [gwr,gtwr]=gauswvlet(dt,fbw,2*tlen);

    % --Creating Radon doamin
    Min = zeros(nq, nt); %initialize Min (radon domain)
    Min_2 = zeros(nq, nt); %Sharper q image

    for ilyr = 1:nlayers
        for jphs = 1: nPhases

            gauswvlt = abs(m(ilyr,jphs)).*stat(gwr,gtwr,tau(ilyr, jphs)); %shift pulses in time. A sample at time t on input will be at t+delt on output

            if m(ilyr, jphs) < 0
                gauswvlt = phsrot(gauswvlt, 180); % rotate this ricker wavelet to negative phase
            end
            indq = find(qs> q(ilyr, jphs), 1);

            % [q(ilyr, jphs), tau(ilyr, jphs), m(ilyr, jphs)]

            %     Min(indq-1, :) = gauswvlt(end-nt+1:end);
            %     Min(indq, :) = gauswvlt(end-nt+1:end);
            %     Min(indq+1, :) = gauswvlt(end-nt+1:end);

            Min_2(indq, :) = gauswvlt(end-nt+1:end);

        end
    end

    mm = max(abs(Min_2), [], 'all'); Min_2 = Min_2 ./ mm;

    % Do for Ps-RF first
    [tx] = forward_radon_freq_umd_Sp(Min_2',dt,rayP,qs,q1,q0,fmin,fmax);
    %[tx] = forward_radon_freq_umd(Min_2',dt,rayP,qs,2, fmin,fmax);
    tx = tx';
    [dx, dy] = size(tx);

    % Plot
    %close all
    twin = [0 25];
    qwin = [-300 300];
    figure(1);
    clf;
    RadonPlot(taus, qs, Min_2', twin, qwin, [], []);

    %[t1, t2, t3] = travelTimesAppx([Vp 8.1], [Vs 4.5], [H 0], rayP, 2, 2);
    [t1, t2, t3] = travelTimesAppx(Vpi, Vsi, Hi, rayP, 2, 2);

    twin = [0 25];
    RFWigglePlot(tx, taus - tshift, rayP, rayP, [t1(end:-1:1) t2(end:-1:1) t3(end:-1:1)], twin, 1, 1, 0);
    title(num2str(isample))
    pause(.2)
end
%% Save RF
% save('Sp_RF_syn.mat', 'tx', 'taus','rayP', 'tshift');