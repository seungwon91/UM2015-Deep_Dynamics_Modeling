
function cascadedLog = log_connector(log_to_expand, log_to_attach)
    cascadedLog = log_to_expand;
    avr_dt = mean(log_to_expand.plotTime(2:end,1)-log_to_expand.plotTime(1:end-1,1));

    cascadedLog.plotTime = [cascadedLog.plotTime; log_to_attach.plotTime+ cascadedLog.plotTime(end,1)+ avr_dt];
    cascadedLog.forwardCommand = [cascadedLog.forwardCommand; log_to_attach.forwardCommand];
    cascadedLog.leftCommand = [cascadedLog.leftCommand; log_to_attach.leftCommand];
    cascadedLog.encoderLeftWheelSpeed = [cascadedLog.encoderLeftWheelSpeed; log_to_attach.encoderLeftWheelSpeed];
    cascadedLog.encoderLeftWheelSpeedSmoothed = [cascadedLog.encoderLeftWheelSpeedSmoothed; log_to_attach.encoderLeftWheelSpeedSmoothed];
    cascadedLog.encoderRightWheelSpeed = [cascadedLog.encoderRightWheelSpeed; log_to_attach.encoderRightWheelSpeed];
    cascadedLog.encoderRightWheelSpeedSmoothed = [cascadedLog.encoderRightWheelSpeedSmoothed; log_to_attach.encoderRightWheelSpeedSmoothed];
    cascadedLog.encoderLinearVelocity = [cascadedLog.encoderLinearVelocity; log_to_attach.encoderLinearVelocity];
    cascadedLog.encoderLinearVelocitySmoothed = [cascadedLog.encoderLinearVelocitySmoothed; log_to_attach.encoderLinearVelocitySmoothed];
    cascadedLog.encoderAngularVelocity = [cascadedLog.encoderAngularVelocity; log_to_attach.encoderAngularVelocity];
    cascadedLog.encoderAngularVelocitySmoothed = [cascadedLog.encoderAngularVelocitySmoothed; log_to_attach.encoderAngularVelocitySmoothed];
    cascadedLog.imuAngularVelocity = [cascadedLog.imuAngularVelocity; log_to_attach.imuAngularVelocity];
end