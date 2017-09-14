%%% wheelchair parameters
Quantum6000Params.wheelCircumference = 1.048; % (m)
Quantum6000Params.wheelBase = 0.591*1.02; % (m) parameter modified to better fit the data
Quantum6000Params.wheelRevolutionPerEncoderPulse = 0.0000046875; % (rev)

%%% motor model parameters - nominal
RobotModelParams.mu = 0.28; % 0.34; % 0.26;  % mechanical energy loss due to rolling friction
% higher alpha can induce higher oscillation
RobotModelParams.beta  = 8.5; % 10; % 5.5;  % motor energy change from velocity
% lower beta induces slower response and higher overshoot
RobotModelParams.gamma = 4.0; % 5.7; %3.49; % 0.165*RobotModelParams.beta/RobotModelParams.mu; % motor energy loss due to resistance, etc
RobotModelParams.alpha = 0.2; % 0.25; % 0.143; % 0.026*RobotModelParams.beta;      % command to input mapping
RobotModelParams.break_disengage_time = 0*0.2; % time until break disengage

%%% controller model parameters - nominal
RobotModelParams.speedRate = 1;
RobotModelParams.turnRate = 0.38; %0.45; %0.315;
RobotModelParams.turnRateInPlace = 044; %0.50; %0.57; % 0.471;
RobotModelParams.turnReductionRate = 0.005;


%%% Parameters fitted against data on carpet Aug 7th 2014
RobotModelParams.mu    = 0.28;
RobotModelParams.beta  = 8.5;
RobotModelParams.gamma = 0.155*RobotModelParams.beta/RobotModelParams.mu;
RobotModelParams.alpha = 0.0245*RobotModelParams.beta;
RobotModelParams.turnRate          = 0.44;
RobotModelParams.turnRateInPlace   = 0.44;
RobotModelParams.turnReductionRate = 0.008;

%%% Parameters fitted against data on stone June 9th 2014
RobotModelParams.mu    = 0.26;
RobotModelParams.beta  = 5.5;
RobotModelParams.gamma = 0.165*RobotModelParams.beta/RobotModelParams.mu;
RobotModelParams.alpha = 0.026*RobotModelParams.beta;
RobotModelParams.turnRate          = 0.45;
RobotModelParams.turnRateInPlace   = 0.50;
RobotModelParams.turnReductionRate = 0.006;

%%% Parameters fitted against data June 17th 2015 from Tufts
if dataType == 'tufts15'
    RobotModelParams.mu = 0.255; % mechanical energy loss due to rolling friction
    % higher alpha can induce higher oscillation
    RobotModelParams.beta  = 5.0; % motor energy change from velocity
    % lower beta induces slower response and higher overshoot
    RobotModelParams.gamma = 0.160*RobotModelParams.beta/RobotModelParams.mu; % motor energy loss due to resistance, etc
    RobotModelParams.alpha = 0.026*RobotModelParams.beta;      % command to input mapping
%     RobotModelParams.break_disengage_time = 0*0.2; % time until break disengage
    RobotModelParams.turnRate          = 0.45;
    RobotModelParams.turnRateInPlace   = 0.45;
    RobotModelParams.turnReductionRate = 0.008;
    
    RobotModelParams.mu    = 0.245;
    RobotModelParams.beta  = 7.0;
    RobotModelParams.gamma = 0.145*RobotModelParams.beta/RobotModelParams.mu;
    RobotModelParams.alpha = 0.0255*RobotModelParams.beta;
    RobotModelParams.turnRate          = 0.455;
    RobotModelParams.turnRateInPlace   = 0.455;
    RobotModelParams.turnReductionRate = 0.008;
end

% %%% quicktests
% RobotModelParams.gamma = 0.168*RobotModelParams.beta/RobotModelParams.alpha;
% RobotModelParams.s     = 0.0255*RobotModelParams.beta;      % command to input mapping
% RobotModelParams.speedRate = 0.95;

% %%% older motor model parameters Apr 17th 2013
% RobotModelParams.alpha = 0.30;  % mechanical energy loss due to rolling friction
% % higher alpha can induce higher oscillation
% RobotModelParams.beta  = 6;     % motor energy change from velocity
% % lower beta induces slower response and higher overshoot
% RobotModelParams.gamma = 0.18*RobotModelParams.beta/RobotModelParams.alpha; % motor energy loss due to resistance, etc
% RobotModelParams.s     = 0.025*RobotModelParams.beta;      % command to input mapping
% RobotModelParams.break_disengage_time = 0.3; % time until break disengage
% 
% %%% controller model parameters
% RobotModelParams.turnRate = 0.32;

% %%% older motor model parameters Apr 12th 2013
% RobotModelParams.alpha = 0.25;  % mechanical energy loss due to rolling friction
% % higher alpha can induce higher oscillation
% RobotModelParams.beta  = 6.5;     % motor energy change from velocity
% % lower beta induces slower response and higher overshoot
% RobotModelParams.gamma = 0.165*RobotModelParams.beta/RobotModelParams.alpha; % motor energy loss due to resistance, etc
% RobotModelParams.s     = 0.025*RobotModelParams.beta;      % command to input mapping
% RobotModelParams.break_disengage_time = 0.3; % time until break disengage
% 
% %%% controller model parameters
% RobotModelParams.turnRate = 0.43;