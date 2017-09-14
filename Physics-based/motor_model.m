function [q, u_friction] = motor_model(q, u_drive, h, mu, beta, gamma, alpha)

    % non-linear rolling friction model for discrete time system
    friction_threshold = -q(1)/h - q(2);
    if abs(friction_threshold) <= mu
        u_friction = friction_threshold;
    else
        u_friction = sign(friction_threshold)*mu;
    end

    % process model
    A = [      1,         h;
         -beta*h, 1-gamma*h];
    B_f  = [   h;
               0];
    B_d  = [       0;
             alpha*h];
         
%     % add saturation
%     u_drive = sign(u_drive)*min(100, abs(u_drive));

    q = A*q + B_f*u_friction + B_d*u_drive;
%     q(2) = sign(q(2))*min(abs(q(2)), 1.5);
end