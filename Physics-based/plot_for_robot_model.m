%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 16.7.8 Updated
% Function to plot time-series prediction comes from the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_for_robot_model(Log, qs, vs, omegas, us_motor, us_joystick, data_range, plot_range, useSmoothed, plotDiff, plotAll)
    
    if nargin < 10
        plotDiff = true;
        plotAll = true;
    end

    figure;
    if plotAll
        subplot(4,1,1)
    else
        subplot(2,1,1)
    end
    hold on
    stairs(Log.plotTime(data_range(1):data_range(2)), Log.encoderLeftWheelSpeed(data_range(1):data_range(2)), 'b');
    if useSmoothed
        stairs(Log.plotTime(data_range(1):data_range(2)), Log.encoderLeftWheelSpeedSmoothed(data_range(1):data_range(2)), 'c');
    end
    stairs(Log.plotTime(data_range(1):data_range(2)), qs(3,:), 'm');
    if plotDiff
        stairs(Log.plotTime(data_range(1):data_range(2)), Log.encoderLeftWheelSpeed(data_range(1):data_range(2))' - qs(3,:), 'r');
    end
    stairs(Log.plotTime(data_range(1):data_range(2)), 0.02*us_motor(2,:), 'k');
    if useSmoothed && plotDiff
        legend('encoder speed','smoothed encoder speed','wheel speed from model','difference','scaled motor command')
    elseif useSmoothed
        legend('encoder speed','smoothed encoder speed','wheel speed from model','scaled motor command')
    elseif plotDiff
        legend('encoder speed','wheel speed from model','difference','scaled motor command')
    else
        legend('encoder speed','wheel speed from model', 'scaled motor command')
    end
    grid minor
    ylim([-2.5 2.5])
    xlim(plot_range)
    title('left motor states');

    if plotAll
        subplot(4,1,2)
    else
        subplot(2,1,2)
    end
    hold on
    stairs(Log.plotTime(data_range(1):data_range(2)), Log.encoderRightWheelSpeed(data_range(1):data_range(2)), 'b');
    if useSmoothed
        stairs(Log.plotTime(data_range(1):data_range(2)), Log.encoderRightWheelSpeedSmoothed(data_range(1):data_range(2)), 'c');
    end
    stairs(Log.plotTime(data_range(1):data_range(2)), qs(1,:), 'm');
    if plotDiff
        stairs(Log.plotTime(data_range(1):data_range(2)), Log.encoderRightWheelSpeed(data_range(1):data_range(2))' - qs(1,:), 'r');
    end
    stairs(Log.plotTime(data_range(1):data_range(2)), 0.02*us_motor(1,:), 'k');
    if useSmoothed && plotDiff
        legend('encoder speed','smoothed encoder speed','wheel speed from model','difference','scaled motor command')
    elseif useSmoothed
        legend('encoder speed','smoothed encoder speed','wheel speed from model','scaled motor command')
    elseif plotDiff
        legend('encoder speed','wheel speed from model','difference','scaled motor command')
    else
        legend('encoder speed','wheel speed from model','scaled motor command')
    end
    grid minor
    ylim([-2.5 2.5])
    xlim(plot_range)
    title('right motor states');

    if plotAll
        subplot(4,1,3)
    else
        figure();
        subplot(2,1,1)
    end
    hold on
    stairs(Log.plotTime(data_range(1):data_range(2)), Log.encoderLinearVelocity(data_range(1):data_range(2)), 'b');
    if useSmoothed
        stairs(Log.plotTime(data_range(1):data_range(2)), Log.encoderLinearVelocitySmoothed(data_range(1):data_range(2)), 'c');
    end
    stairs(Log.plotTime(data_range(1):data_range(2)), vs, 'm');
    if plotDiff
        stairs(Log.plotTime(data_range(1):data_range(2)), Log.encoderLinearVelocity(data_range(1):data_range(2))' - vs, 'r');
    end
    stairs(Log.plotTime(data_range(1):data_range(2)), 0.02*us_joystick(1,:), 'k');
    if useSmoothed && plotDiff
        legend('linear velocity from data','smoothed linear velocity','linear velocity from model','difference','scaled velocity command')
    elseif useSmoothed
        legend('linear velocity from data','smoothed linear velocity','linear velocity from model','scaled velocity command')
    elseif plotDiff
        legend('linear velocity from data','linear velocity from model','difference','scaled velocity command')
    else
        legend('linear velocity from data','linear velocity from model','scaled velocity command')
    end
    grid minor
    ylim([-0.5 1.5])
    xlim(plot_range)
    title('linear velocity');

    if plotAll
        subplot(4,1,4)
    else
        subplot(2,1,2)
    end
    hold on
    stairs(Log.plotTime(data_range(1):data_range(2)), Log.encoderAngularVelocity(data_range(1):data_range(2)), 'b');
    if useSmoothed
        stairs(Log.plotTime(data_range(1):data_range(2)), Log.encoderAngularVelocitySmoothed(data_range(1):data_range(2)), 'c');
    end
    stairs(Log.plotTime(data_range(1):data_range(2)), omegas, 'm');
    if plotDiff
        stairs(Log.plotTime(data_range(1):data_range(2)), Log.encoderAngularVelocity(data_range(1):data_range(2))' - omegas, 'r');
    end
    stairs(Log.plotTime(data_range(1):data_range(2)), 0.02*us_joystick(2,:), 'k');
    if useSmoothed && plotDiff
        legend('angular velocity from data','smoothed angular velocity','angular velocity from model','difference','scaled joystick command')
    elseif useSmoothed
        legend('angular velocity from data','smoothed angular velocity','angular velocity from model','scaled joystick command')
    elseif plotDiff
        legend('angular velocity from data','angular velocity from model','difference','scaled joystick command')
    else
        legend('angular velocity from data','angular velocity from model','scaled joystick command')
    end
    grid minor
    ylim([-2.5 2.5])
    xlim(plot_range)
    title('angular velocity');
end