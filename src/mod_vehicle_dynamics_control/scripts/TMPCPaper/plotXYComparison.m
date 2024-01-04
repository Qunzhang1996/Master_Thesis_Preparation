function plotXYComparison(files, laps, labels)
% Author:       Alexander Wischnewski
% Description:  
%   function used to compare different controllers 
% Inputs/parameters:
%   files:      Cell array with file names of logs
%   laps:       laps to evaluate (matrix with start and end lap for each datafile)
%   labels:     Cell array with abels for the datasets used for legends 
%               (file names are used if not given) 

% plotting parameters
LineWidth = 1; 
colors = {[0, 0.3961, 0.7412], [0.8902,0.44706,0.1333], [0.6353,0.6784,0]};
sLim_lower = 250; 
sLim_upper = 700; 

% load all the relevant data files
for i = 1:1:length(files) 
    data{i} = load(files{i}); 
end

% check if labels are given, if not use file names
if(nargin <= 2) 
    labels = files; 
end

% find start and end indices
for i = 1:1:length(files) 
    idx_start{i} = find((data{i}.debug.debug_mvdc_path_matching_debug_ActualTrajPoint_LapCnt.Data == laps(i,1)), 1, 'first') + 50; 
    idx_end{i} = find((data{i}.debug.debug_mvdc_path_matching_debug_ActualTrajPoint_LapCnt.Data == laps(i,2)), 1, 'last') - 50; 
end

figure; 
hold on; grid on; box on; 
plot(data{i}.debug.debug_mvdc_path_matching_debug_ActualTrajPoint_x_m.Data(idx_start{i}:idx_end{i}), ...
    data{i}.debug.debug_mvdc_path_matching_debug_ActualTrajPoint_y_m.Data(idx_start{i}:idx_end{i}) + 1, ...
    'LineWidth', LineWidth, 'Color', 'k', 'LineStyle', '--', 'DisplayName', 'Target Path');  
plot(data{i}.debug.debug_mvdc_path_matching_debug_ActualTrajPoint_x_m.Data(idx_start{i}:idx_end{i}), ...
    data{i}.debug.debug_mvdc_path_matching_debug_ActualTrajPoint_y_m.Data(idx_start{i}:idx_end{i}) - 1, ...
    'LineWidth', LineWidth, 'Color', 'k', 'LineStyle', '--', 'HandleVisibility', 'off'); 
for i = 1:1:length(files) 
    plot(data{i}.debug.debug_mvdc_state_estimation_debug_StateEstimate_Pos_x_m.Data(idx_start{i}:idx_end{i}), ...
        data{i}.debug.debug_mvdc_state_estimation_debug_StateEstimate_Pos_y_m.Data(idx_start{i}:idx_end{i}), ...
        'LineWidth', LineWidth, 'Color', colors{i}, 'DisplayName', [labels{i}]); 
end
xlabel('x East in m'); ylabel({'y North in m'}); % ylim([0, 1.1]);xlim([sLim_lower, sLim_upper]); 
legend();
matlab2tikz('XYComparison.tex', 'standalone', true);