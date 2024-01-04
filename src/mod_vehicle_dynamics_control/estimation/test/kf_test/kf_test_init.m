%% Documentation       
%
% Authors:      Alexander Wischnewski (alexander.wischnewski@tum.de)
%               
% Start Date:   14.06.2019
% 
% Description:  loads a log data file and converts that to the required bus
%               structures for measurement data replay with the model
%               kf_test.slx

%% Script
% Dataset to load 
data = load('data/20_19_28/SimTest3.mat'); 

% dummy timeseries for non-required signals 
ts_zero = timeseries(zeros(length(data.debug.debug_Time_s.Time), 1), ...
    data.debug.debug_Time_s.Time); 

% assemble VehicleSensorData bus 
in1.Delta_Manual_rad = data.debug.debug_VehicleSensorData_Delta_Manual_rad; 
in1.Throttle_Manual_perc = data.debug.debug_VehicleSensorData_Throttle_Manual_perc; 
in1.Brake_Manual_bar = data.debug.debug_VehicleSensorData_Brake_Manual_bar; 
in1.Gear_Manual = data.debug.debug_VehicleSensorData_Gear_Manual; 
in1.T_CoolantTemp_Celsius = data.debug.debug_VehicleSensorData_T_CoolantTemp_Celsius; 
in1.T_TransmissionOilTemp_Celsius = data.debug.debug_VehicleSensorData_T_TransmissionOilTemp_Celsius; 
in1.p_Fuel_kPa = data.debug.debug_VehicleSensorData_p_Fuel_kPa; 
in1.p_EngineOil_kPa = data.debug.debug_VehicleSensorData_p_EngineOil_kPa; 
in1.T_Engine_Nm = data.debug.debug_VehicleSensorData_T_Engine_Nm; 
in1.GearEngaged = timeseries(int16(data.debug.debug_VehicleSensorData_GearEngaged.Data), data.debug.debug_VehicleSensorData_GearEngaged.Time); 
in1.omega_Engine_radps = data.debug.debug_VehicleSensorData_omega_Engine_radps; 
in1.ThrottlePos_perc = data.debug.debug_VehicleSensorData_ThrottlePos_perc; 
in1.T_WheelFL_Nm = data.debug.debug_VehicleSensorData_T_WheelFL_Nm; 
in1.T_WheelFR_Nm = data.debug.debug_VehicleSensorData_T_WheelFR_Nm; 
in1.T_WheelRL_Nm = data.debug.debug_VehicleSensorData_T_WheelRL_Nm; 
in1.T_WheelRR_Nm = data.debug.debug_VehicleSensorData_T_WheelRR_Nm; 
in1.Delta_Wheel_rad = data.debug.debug_VehicleSensorData_Delta_Wheel_rad; 
in1.p_BrakeF_bar = data.debug.debug_VehicleSensorData_p_BrakeF_bar; 
in1.p_BrakeR_bar = data.debug.debug_VehicleSensorData_p_BrakeR_bar; 
in1.omega_WheelFL_radps = data.debug.debug_VehicleSensorData_omega_WheelFL_radps; 
in1.omega_WheelFR_radps = data.debug.debug_VehicleSensorData_omega_WheelFR_radps; 
in1.omega_WheelRL_radps = data.debug.debug_VehicleSensorData_omega_WheelRL_radps; 
in1.omega_WheelRR_radps = data.debug.debug_VehicleSensorData_omega_WheelRR_radps; 
in1.valid_Wheelspeeds_b = timeseries(boolean(data.debug.debug_VehicleSensorData_valid_Wheelspeeds_b.Data), data.debug.debug_VehicleSensorData_valid_Wheelspeeds_b.Time); 
in1.x_Loc1_m = data.debug.debug_VehicleSensorData_x_Loc1_m; 
in1.y_Loc1_m = data.debug.debug_VehicleSensorData_y_Loc1_m;
in1.z_Loc1_m = data.debug.debug_VehicleSensorData_z_Loc1_m;
in1.phi_RollAngleLoc1_rad = data.debug.debug_VehicleSensorData_phi_RollAngleLoc1_rad;
in1.theta_PitchAngleLoc1_rad = data.debug.debug_VehicleSensorData_theta_PitchAngleLoc1_rad;
in1.psi_YawAngleLoc1_rad = data.debug.debug_VehicleSensorData_psi_YawAngleLoc1_rad;
in1.t_EstimateLoc1_s = data.debug.debug_VehicleSensorData_t_EstimateLoc1_s;
in1.valid_Loc1_b = timeseries(boolean(data.debug.debug_VehicleSensorData_valid_Loc1_b.Data), data.debug.debug_VehicleSensorData_valid_Loc1_b.Time); 
in1.accuracy_Loc1 = LocEmu;
in1.x_Loc2_m = data.debug.debug_VehicleSensorData_x_Loc2_m;
in1.y_Loc2_m = data.debug.debug_VehicleSensorData_y_Loc2_m;
in1.z_Loc2_m = data.debug.debug_VehicleSensorData_z_Loc2_m;
in1.phi_RollAngleLoc2_rad = data.debug.debug_VehicleSensorData_phi_RollAngleLoc2_rad;
in1.theta_PitchAngleLoc2_rad = data.debug.debug_VehicleSensorData_theta_PitchAngleLoc2_rad;
in1.psi_YawAngleLoc2_rad = data.debug.debug_VehicleSensorData_psi_YawAngleLoc2_rad;
in1.t_EstimateLoc2_s = data.debug.debug_VehicleSensorData_t_EstimateLoc2_s;
in1.valid_Loc2_b = timeseries(boolean(data.debug.debug_VehicleSensorData_valid_Loc2_b.Data), data.debug.debug_VehicleSensorData_valid_Loc2_b.Time); 
in1.accuracy_Loc2 = LocEmu;
in1.x_Loc3_m = data.debug.debug_VehicleSensorData_x_Loc3_m;
in1.y_Loc3_m = data.debug.debug_VehicleSensorData_y_Loc3_m;
in1.z_Loc3_m = data.debug.debug_VehicleSensorData_z_Loc3_m;
in1.phi_RollAngleLoc3_rad = data.debug.debug_VehicleSensorData_phi_RollAngleLoc3_rad;
in1.theta_PitchAngleLoc3_rad = data.debug.debug_VehicleSensorData_theta_PitchAngleLoc3_rad;
in1.psi_YawAngleLoc3_rad = data.debug.debug_VehicleSensorData_psi_YawAngleLoc3_rad;
in1.t_EstimateLoc3_s = data.debug.debug_VehicleSensorData_t_EstimateLoc3_s;
in1.valid_Loc3_b = timeseries(boolean(data.debug.debug_VehicleSensorData_valid_Loc3_b.Data), data.debug.debug_VehicleSensorData_valid_Loc3_b.Time); 
in1.accuracy_Loc3 = LocEmu;
in1.vx_CoGVel1_mps = data.debug.debug_VehicleSensorData_vx_CoGVel1_mps;
in1.vy_CoGVel1_mps = data.debug.debug_VehicleSensorData_vy_CoGVel1_mps;
in1.valid_Vel1_b = timeseries(boolean(data.debug.debug_VehicleSensorData_valid_Vel1_b.Data), data.debug.debug_VehicleSensorData_valid_Vel1_b.Time); 
in1.vx_CoGVel2_mps = data.debug.debug_VehicleSensorData_vx_CoGVel2_mps;
in1.vy_CoGVel2_mps = data.debug.debug_VehicleSensorData_vy_CoGVel2_mps;
in1.valid_Vel2_b = timeseries(boolean(data.debug.debug_VehicleSensorData_valid_Vel2_b.Data), data.debug.debug_VehicleSensorData_valid_Vel2_b.Time); 
in1.ax_CoGIMU1_mps2 = data.debug.debug_VehicleSensorData_ax_CoGIMU1_mps2;
in1.ay_CoGIMU1_mps2 = data.debug.debug_VehicleSensorData_ay_CoGIMU1_mps2;
in1.az_CoGIMU1_mps2 = data.debug.debug_VehicleSensorData_az_CoGIMU1_mps2;
in1.dPhi_RollRateIMU1_radps = data.debug.debug_VehicleSensorData_dPhi_RollRateIMU1_radps;
in1.dTheta_PitchRateIMU1_radps = data.debug.debug_VehicleSensorData_dTheta_PitchRateIMU1_radps;
in1.dPsi_YawRateIMU1_radps = data.debug.debug_VehicleSensorData_dPsi_YawRateIMU1_radps;
in1.valid_IMU1_b = timeseries(boolean(data.debug.debug_VehicleSensorData_valid_IMU1_b.Data), data.debug.debug_VehicleSensorData_valid_IMU1_b.Time); 
in1.ax_CoGIMU2_mps2 = data.debug.debug_VehicleSensorData_ax_CoGIMU2_mps2;
in1.ay_CoGIMU2_mps2 = data.debug.debug_VehicleSensorData_ay_CoGIMU2_mps2;
in1.az_CoGIMU2_mps2 = data.debug.debug_VehicleSensorData_az_CoGIMU2_mps2;
in1.dPhi_RollRateIMU2_radps = data.debug.debug_VehicleSensorData_dPhi_RollRateIMU2_radps;
in1.dTheta_PitchRateIMU2_radps = data.debug.debug_VehicleSensorData_dTheta_PitchRateIMU2_radps;
in1.dPsi_YawRateIMU2_radps = data.debug.debug_VehicleSensorData_dPsi_YawRateIMU2_radps;
in1.valid_IMU2_b = timeseries(boolean(data.debug.debug_VehicleSensorData_valid_IMU2_b.Data), data.debug.debug_VehicleSensorData_valid_IMU2_b.Time); 

% assemble enable signal
EnableSensorFusion = data.debug.debug_mloc_statemachine_debug_EnableSensorFusion; 

% assemble final bus signals and input structure
input = Simulink.SimulationData.Dataset;
input = input.addElement(in1,'VehicleSensorData');  
input = input.addElement(EnableSensorFusion, 'EnableSensorFusion');   
input = input.addElement(data.debug.debug_mvdc_state_estimation_debug_StateEstimate_vx_mps, 'vx_Real'); 
input = input.addElement(data.debug.debug_mvdc_state_estimation_debug_StateEstimate_vy_mps, 'vy_Real'); 
input = input.addElement(data.debug.debug_mvdc_state_estimation_debug_Residuals_res_dLoc1_m, 'resd_Real');