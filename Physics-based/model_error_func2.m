function [err, vs, omegas, us_joystick, qs, us_motor, fs] = model_error_func2(x, steadyParams, wheelbase, Log, dataBegin, dataEnd, useSmoothed)

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

    turnRateR          = x(5);
    turnReductionRateR = x(6);
    turnRateL          = x(5);
    turnReductionRateL = x(6);
elseif length(x) == 7 && isempty(steadyParams)
    mu                = x(1);
    beta              = x(2);
    gamma             = x(3);
    alpha             = x(4);

    turnRateR          = x(5);
    turnReductionRateR = x(6);
    turnRateInPlaceR   = x(7);
    turnRateL          = x(5);
    turnReductionRateL = x(6);
    turnRateInPlaceL   = x(7);
elseif length(x) == 8 && isempty(steadyParams)
    mu                = x(1);
    beta              = x(2);
    gamma             = x(3);
    alpha             = x(4);

    turnRateR          = x(5);
    turnReductionRateR = x(6);
    turnRateL          = x(7);
    turnReductionRateL = x(8);
elseif length(x) == 9 && isempty(steadyParams)
    mu                = x(1);
    beta              = x(2);
    gamma             = x(3);
    alpha             = x(4);

    turnRateR          = x(5);
    turnReductionRateR = x(6);
    turnRateInPlaceR   = x(9);
    turnRateL          = x(7);
    turnReductionRateL = x(8);
    turnRateInPlaceL   = x(9);
elseif length(x) == 10 && isempty(steadyParams)
    muR                = x(1);
    betaR              = x(2);
    gammaR             = x(3);
    alphaR             = x(4);
    muL                = x(5);
    betaL              = x(6);
    gammaL             = x(7);
    alphaL             = x(8);

    turnRateR          = x(9);
    turnReductionRateR = x(10);
    turnRateL          = x(9);
    turnReductionRateL = x(10);
elseif length(x) == 11 && isempty(steadyParams)
    muR                = x(1);
    betaR              = x(2);
    gammaR             = x(3);
    alphaR             = x(4);
    muL                = x(5);
    betaL              = x(6);
    gammaL             = x(7);
    alphaL             = x(8);

    turnRateR          = x(9);
    turnReductionRateR = x(10);
    turnRateInPlaceR   = x(11);
    turnRateL          = x(9);
    turnReductionRateL = x(10);
    turnRateInPlaceL   = x(11);
elseif length(x) == 12 && isempty(steadyParams)
    muR                = x(1);
    betaR              = x(2);
    gammaR             = x(3);
    alphaR             = x(4);
    muL                = x(5);
    betaL              = x(6);
    gammaL             = x(7);
    alphaL             = x(8);

    turnRateR          = x(9);
    turnReductionRateR = x(10);
    turnRateL          = x(11);
    turnReductionRateL = x(12);
elseif length(x) == 13 && isempty(steadyParams)
    muR                = x(1);
    betaR              = x(2);
    gammaR             = x(3);
    alphaR             = x(4);
    muL                = x(5);
    betaL              = x(6);
    gammaL             = x(7);
    alphaL             = x(8);

    turnRateR          = x(9);
    turnReductionRateR = x(10);
    turnRateInPlaceR = x(13);
    turnRateL          = x(11);
    turnReductionRateL = x(12);
    turnRateInPlaceL = x(13);
elseif length(x) == 4 && ~isempty(steadyParams)
    mu                = x(1);
    beta              = x(2);
    gamma             = x(3);
    alpha             = x(4);
    
    if length(steadyParams) == 2
        turnRateR          = steadyParams(1);
        turnReductionRateR = steadyParams(2);
        turnRateInPlaceR = steadyParams(1);
        turnRateL          = steadyParams(1);
        turnReductionRateL = steadyParams(2);
        turnRateInPlaceL = steadyParams(1);
    elseif length(steadyParams) == 3
        turnRateR          = steadyParams(1);
        turnReductionRateR = steadyParams(2);
        turnRateInPlaceR = steadyParams(3);
        turnRateL          = steadyParams(1);
        turnReductionRateL = steadyParams(2);
        turnRateInPlaceL = steadyParams(3);
    elseif length(steadyParams) == 4
        turnRateR          = steadyParams(1);
        turnReductionRateR = steadyParams(2);
        turnRateInPlaceR = steadyParams(1);
        turnRateL          = steadyParams(3);
        turnReductionRateL = steadyParams(4);
        turnRateInPlaceL = steadyParams(3);
    elseif length(steadyParams) == 5
        turnRateR          = steadyParams(1);
        turnReductionRateR = steadyParams(2);
        turnRateInPlaceR = steadyParams(5);
        turnRateL          = steadyParams(3);
        turnReductionRateL = steadyParams(4);
        turnRateInPlaceL = steadyParams(5);
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
    
    if length(steadyParams) == 2
        turnRateR          = steadyParams(1);
        turnReductionRateR = steadyParams(2);
        turnRateL          = steadyParams(1);
        turnReductionRateL = steadyParams(2);
    elseif length(steadyParams) == 4
        turnRateR          = steadyParams(1);
        turnReductionRateR = steadyParams(2);
        turnRateL          = steadyParams(3);
        turnReductionRateL = steadyParams(4);
    end
end
speedRate         = 1;

% initial motor and break states
q_r = [0;0];
q_l = [0;0];

% storage variables
us_joystick = zeros(2, dataEnd);
us_motor    = zeros(2, dataEnd);
fs          = zeros(2, dataEnd);
qs          = zeros(4, dataEnd);
vs          = zeros(1, dataEnd);
omegas      = zeros(1, dataEnd);

for i = dataBegin:(dataEnd - 1);
    % current time
    t = Log.plotTime(i);
    dt = Log.plotTime(i+1) - t; % in simulation this should be a fixed number. Here it is changing to match data.
    
    % current command
    u_joystick = [Log.forwardCommand(i); Log.leftCommand(i)];
    
    % convert joystick command to individual motor commands
%     turnRateModified = turnRate*(1-turnReductionRate*u_joystick(1));
%     if u_joystick(1) == 0
%         u_motor = [1 turnRateInPlace; 1 -turnRateInPlace]*u_joystick;
%     else
%         u_motor = [speedRate turnRateModified; speedRate -turnRateModified]*u_joystick;
%     end
    if u_joystick(1) == 0
        if exist('turnRateInPlaceR','var')
            turnRateModified_r = turnRateInPlaceR;
            turnRateModified_l = turnRateInPlaceL;
        else
            turnRateModified_r = turnRateR;
            turnRateModified_l = turnRateL;
        end
    else
        turnRateModified_r = turnRateR*(1-turnReductionRateR*u_joystick(1));
        turnRateModified_l = turnRateL*(1-turnReductionRateL*u_joystick(1));
    end
    u_motor = [speedRate turnRateModified_r; speedRate -turnRateModified_l]*u_joystick;
  
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
    us_joystick(:,i+1) = u_joystick;
    us_motor(:,i+1)    = u_motor;
    fs(:,i+1)          = [u_friction_r; u_friction_l];
    qs(:,i+1)          = [q_r; q_l];
    vs(:,i+1)          = v;
    omegas(:,i+1)      = omega;
end

motorSaturationThresh = 2.0;
if useSmoothed
    left_v_diff = Log.encoderLeftWheelSpeedSmoothed(dataBegin:dataEnd,1)' - qs(3, dataBegin:dataEnd);
    right_v_diff = Log.encoderRightWheelSpeedSmoothed(dataBegin:dataEnd,1)' - qs(1, dataBegin:dataEnd);
    v_diff     = Log.encoderLinearVelocitySmoothed(dataBegin:dataEnd,1)'  - vs(1, dataBegin:dataEnd);
    omega_diff = Log.encoderAngularVelocitySmoothed(dataBegin:dataEnd,1)' - omegas(1, dataBegin:dataEnd);
else
    left_v_diff = Log.encoderLeftWheelSpeed(dataBegin:dataEnd,1)' - qs(3, dataBegin:dataEnd);
    right_v_diff = Log.encoderRightWheelSpeed(dataBegin:dataEnd,1)' - qs(1, dataBegin:dataEnd);
    v_diff     = Log.encoderLinearVelocity(dataBegin:dataEnd,1)'  - vs(1, dataBegin:dataEnd);
    omega_diff = Log.encoderAngularVelocity(dataBegin:dataEnd,1)' - omegas(1, dataBegin:dataEnd);
end

vs = vs(:,dataBegin:dataEnd);
omegas = omegas(:,dataBegin:dataEnd);
us_joystick = us_joystick(:,dataBegin:dataEnd);
qs = qs(:,dataBegin:dataEnd);
us_motor = us_motor(:,dataBegin:dataEnd);
fs = fs(:,dataBegin:dataEnd);


rightWheelOverCount = abs(qs(2,:)) > motorSaturationThresh;
leftWheelOverCount  = abs(qs(4,:)) > motorSaturationThresh;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error Type Setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rms_err = sum(v_diff.^2 + omega_diff.^2);
% rms_err = sum(abs(v_diff) + abs(omega_diff));
% rms_err = sum(abs(left_v_diff) + abs(right_v_diff));

% steady_state_error = model_steady_state_error_func(Log, qs, vs, omegas, useSmoothed, false);
% trans_error = model_transition_error_func(Log, qs, vs, omegas, useSmoothed);
% trans_to_steady_ratio = 0;
% % rms_err = sum(steady_state_error.linear.^2 + steady_state_error.angular.^2);
% % rms_err = sum(steady_state_error.left_wheel.^2 + steady_state_error.right_wheel.^2);
% rms_err = sum(steady_state_error.linear.^2 + steady_state_error.angular.^2 + trans_to_steady_ratio*(trans_error.linear.^2 + trans_error.angular.^2));

% history of errors(NOT YET USED)
% rms_err = sqrt(sum(v_diff.^2 + omega_diff.^2));
% rms_err = sum(v_diff.^4 + omega_diff.^4);
% rms_err = sqrt(sum(abs(v_diff))^2 + sum(abs(omega_diff))^2);
% rms_err = sum(abs(v_diff))^2 + sum(abs(omega_diff))^2;
% rms_err = sum(abs(v_diff) + abs(omega_diff));
% rms_err = sum(exp(-((v_diff - 0.5).^2)./((0.5)^2)) + exp(-((omega_diff - 0.5).^2)./((0.5)^2)));
% rms_err = sum(exp(sqrt(v_diff.^2 + omega_diff.^2)));
penalty = 1000*(sum(leftWheelOverCount) + sum(rightWheelOverCount))/dataEnd;
err = rms_err + 0*penalty;

disp(['iteration ', num2str(iterations)]);
toc();
disp(['rms_err = ',      num2str(rms_err), ...
      ',    penalty = ', num2str(penalty), ...
      ',    err = ',     num2str(err)]);

end