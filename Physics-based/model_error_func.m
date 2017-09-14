function [err, vs, omegas, us_joystick, qs, us_motor, fs] = model_error_func(x, steadyParams, wheelbase, data_input, data_output)

    persistent iterations;
    if(isempty(iterations))
        iterations = 1;
    else
        iterations = iterations + 1;
    end

    tic();

    % parameters
    if length(x) == 6 && isempty(steadyParams)
        mu                = x(1);
        beta              = x(2);
        gamma             = x(3);
        alpha             = x(4);

        turnRate          = x(5);
        turnRateInPlace   = x(5);
        turnReductionRate = x(6);
    elseif length(x) == 7 && isempty(steadyParams)
        mu                = x(1);
        beta              = x(2);
        gamma             = x(3);
        alpha             = x(4);

        turnRate          = x(5);
        turnRateInPlace   = x(6);
        turnReductionRate = x(7);
    elseif length(x) == 10 && isempty(steadyParams)
        muR                = x(1);
        betaR              = x(2);
        gammaR             = x(3);
        alphaR             = x(4);
        muL                = x(5);
        betaL              = x(6);
        gammaL             = x(7);
        alphaL             = x(8);

        turnRate          = x(9);
        turnRateInPlace   = x(9);
        turnReductionRate = x(10);
    elseif length(x) == 11 && isempty(steadyParams)
        muR                = x(1);
        betaR              = x(2);
        gammaR             = x(3);
        alphaR             = x(4);
        muL                = x(5);
        betaL              = x(6);
        gammaL             = x(7);
        alphaL             = x(8);

        turnRate          = x(9);
        turnRateInPlace   = x(10);
        turnReductionRate = x(11);
    elseif length(x) == 4 && ~isempty(steadyParams)
        mu                = x(1);
        beta              = x(2);
        gamma             = x(3);
        alpha             = x(4);

        turnRate          = steadyParams(1);
        if length(steadyParams) == 2
            turnRateInPlace   = steadyParams(1);
            turnReductionRate = steadyParams(2);
        elseif length(steadyParams) == 3
            turnRateInPlace   = steadyParams(2);
            turnReductionRate = steadyParams(3);
        end
    elseif length(x) == 8 && ~isempty(steadyParams)
        muR                = x(1);
        betaR              = x(2);
        gammaR             = x(3);
        alphaR             = x(4);
        muL                = x(5);
        betaL              = x(6);
        gammaL             = x(7);
        alphaL             = x(8);

        turnRate          = steadyParams(1);
        if length(steadyParams) == 2
            turnRateInPlace   = steadyParams(1);
            turnReductionRate = steadyParams(2);
        elseif length(steadyParams) == 3
            turnRateInPlace   = steadyParams(2);
            turnReductionRate = steadyParams(3);
        end
    end
    speedRate         = 1;

    num_data = size(data_input, 1);
    num_pred_step = size(data_output, 2)/2;

    % storage variables
    us_joystick = zeros(num_data, 2*num_pred_step);
    us_motor    = zeros(num_data, 2*num_pred_step);
    fs          = zeros(num_data, 2*num_pred_step);
    qs          = zeros(num_data, 2*num_pred_step);
    vs          = zeros(num_data, num_pred_step);
    omegas      = zeros(num_data, num_pred_step);

    for data_cnt = 1:num_data
        % initial motor and break states
        q_l = data_input(data_cnt, 1:2)';
        q_r = data_input(data_cnt, 3:4)';

        dt = 1/25;
        for pred_cnt = 1:num_pred_step
            % current command
            u_joystick = [data_input(data_cnt, 3+2*pred_cnt); data_input(data_cnt, 4+2*pred_cnt)];

            % convert joystick command to individual motor commands
            turnRateModified = turnRate*(1-turnReductionRate*u_joystick(1));
            if u_joystick(1) == 0
                u_motor = [1 turnRateInPlace; 1 -turnRateInPlace]*u_joystick;
            else
                u_motor = [speedRate turnRateModified; speedRate -turnRateModified]*u_joystick;
            end

            % run right and left motor model (now symmetric)
            if exist('muL','var')
                [q_r, u_friction_r] = motor_model(q_r, u_motor(1), dt, muR, betaR, gammaR, alphaR);
                [q_l, u_friction_l] = motor_model(q_l, u_motor(2), dt, muL, betaL, gammaL, alphaL);
            else
                [q_r, u_friction_r] = motor_model(q_r, u_motor(1), dt, mu, beta, gamma, alpha);
                [q_l, u_friction_l] = motor_model(q_l, u_motor(2), dt, mu, beta, gamma, alpha);
            end

            % convert motor speeds to linear and angular velocities
            v     = (q_r(1) + q_l(1))*0.5;
            omega = (q_r(1) - q_l(1))/wheelbase;

            % save data
            us_joystick(data_cnt, 2*pred_cnt-1:2*pred_cnt) = u_joystick;
            us_motor(data_cnt, 2*pred_cnt-1:2*pred_cnt)    = u_motor;
            fs(data_cnt, 2*pred_cnt-1:2*pred_cnt)          = [u_friction_l; u_friction_r];
            qs(data_cnt, 2*pred_cnt-1:2*pred_cnt)          = [q_l(1); q_r(1)];
            vs(data_cnt, pred_cnt)                         = v;
            omegas(data_cnt, pred_cnt)                     = omega;
        end

%         if mod(data_cnt, 5000) == 0
%             fprintf(1, '\t%d', data_cnt);
%         end
    end

    wheel_v_diff = data_output - qs;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error Type Setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mae_err = sum(sum(abs(wheel_v_diff)));
    rms_err = sum(sum(wheel_v_diff.^2));
    max_err = max(max(abs(wheel_v_diff)));

    %%%% penalty part(not modified yet)
    % motorSaturationThresh = 2.0;
    % rightWheelOverCount = abs(qs(2,:)) > motorSaturationThresh;
    % leftWheelOverCount  = abs(qs(4,:)) > motorSaturationThresh;
    % penalty = 1000*(sum(leftWheelOverCount) + sum(rightWheelOverCount))/dataEnd;

    penalty = 0;
%     err = rms_err + 0*penalty;
    err = [mae_err; rms_err; max_err];

    disp(['iteration ', num2str(iterations)]);
    toc();
%     disp(['rms_err = ',      num2str(rms_err), ...
%           ',    penalty = ', num2str(penalty), ...
%           ',    err = ',     num2str(err)]);
end