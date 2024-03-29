function exportPredictionDataCSV(filename, debug)
%
% Author: Alexander Wischnewski     Date: 07-02-2020
% 
% Description: 
%   exports the data required for disturbance prediction to csv
% 
% Input: 
%   filename: Output filename
%   debug: Debug structure

M = [debug.debug_mvdc_path_feedback_debug_Lat_ay_Target_mps2.Time, ...
    debug.debug_mvdc_path_feedback_debug_Lat_ay_Target_mps2.Data, ...
    debug.debug_mvdc_path_feedback_debug_Lat_DistEstimate_mps2.Data, ...
    debug.debug_mvdc_path_matching_debug_ActualPathPoint_v_mps.Data, ...
    debug.debug_mvdc_path_matching_debug_ActualPathPoint_kappa_radpm.Data, ...
    debug.debug_mvdc_path_feedback_debug_Lat_LatPosError_m.Data, ...
    debug.debug_mvdc_curvvel_tracking_debug_Curv_TargetAy_mps2.Data, ...
    debug.debug_mvdc_path_feedback_debug_Lat_LatPosError_DerAna_mps.Data]; 

csvwrite(filename, M)
