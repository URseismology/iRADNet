function [tau, q, q1, q0] = get_q_t_Sp(H, Vp, Vs)
    % function [tau, q, q1, q0] = get_q_t_Sp(H, Vp, Vs)
    %
    % Author: Ziqi Zhang
    %
    % Input: H (Thickness), Vp, Vs as row vectors. Only two layers allowed.
    %
    % Both q and tau are 1 (interface) by 3 (phase) matrices.
    % Both q1 and q0 are scalars (order-1 and order-0 polyfit parameters for the second multiple).

    %% Parameter setup

    rayP = (0.01:0.01:0.135); % for Sp （0.098:0.001:0.135)

    %% Calculate q & tau

    [t1, t2, t3] = travelTimesAppx(Vs, Vp, H, rayP, 1, 2); % Vp and Vs reversed for Sp

    fit_param_1 = polyfit(rayP', t1, 2);
    fit_param_2 = polyfit(rayP', t2, 2);
    fit_param_3 = polyfit(rayP', t3, 2);

    tau = [fit_param_1(3) fit_param_2(3) fit_param_3(3)];
    q = [fit_param_1(1) fit_param_2(1) fit_param_3(1)];

    q1 = fit_param_3(2);
    q0 = fit_param_3(3);
