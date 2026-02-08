clear all; clc; close all; 

set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',12)

% include yaml library to read the configuration file 
addpath("+yaml")
cm = [0.5 0.4 1; 1 1 1; 0.9961 0.3804 0 ]; % [0 0 1; 1 1 1; 1 0 0]; 
cmi = interp1([-5; 0; 5], cm, (-5:0.1:5)); 
%% LOAD DATA 

output_folder = "../output";
test_folder = "snr10_pb132"; 
types = ["noise_agnostic", "noise_aware"];
loss_id = [1, 2, 3];
yml_file = "optimization_real_data_gpu.yml";

fullfilepath = fullfile(output_folder, test_folder, types(1));
dirinfo = dir(fullfilepath);
dirinfo = dirinfo([dirinfo.isdir]);
dirnames = {dirinfo.name};
run_dir = dirnames(~ismember(dirnames, {'.', '..'}));
rirs_id  = [1]; %[1, 3, 5, 8, 9, 11]; 
n_rirs = length(rirs_id);

fullfilepath = fullfile(output_folder, test_folder, types(1), run_dir{1});
dirinfo = dir(fullfilepath);
dirinfo = dirinfo([dirinfo.isdir]);
dirnames = {dirinfo.name};
loss_dir = dirnames(~ismember(dirnames, {'.', '..'}));

i_loss = 2; 
rir_filepath = fullfile("..", "real_rirs", "sdm_selected");

%% Define colors
color_agn = [0.2, 0.4, 0.8];      % Blue
color_awa = [0.9, 0.3, 0.2];      % Red
color_gt = [0.2, 0.2, 0.2];       % Dark gray

%%

for i_rir=1:n_rirs
    rir_name = strrep(run_dir{rirs_id(i_rir)}, '_', ' ') + " " + loss_dir{i_loss};
    
    % Create figure with better size and position
    figure('Name', rir_name, 'NumberTitle', 'off', ...
           'Position', [100, 100, 1200, 450], ...
           'Color', 'w');
    
    % Load targets of the rir 
    analysis_file = strrep(run_dir{rirs_id(i_rir)}, 'target_', '') + "_analysis.mat";
    rir_analysis = load(fullfile(rir_filepath, analysis_file));
 
    curr_dir = fullfile(output_folder, test_folder, types(1), run_dir{rirs_id(i_rir)}, loss_dir{i_loss});
    fdn_analysis_agn = load(fullfile(curr_dir, "optimized_rir_analysis.mat"));
    curr_dir = fullfile(output_folder, test_folder, types(2), run_dir{rirs_id(i_rir)}, loss_dir{i_loss});
    fdn_analysis_awa = load(fullfile(curr_dir, "optimized_rir_analysis.mat"));

    % Subplot 1: Amplitude
    subplot(1, 2, 1)    
    h1 = plot(fdn_analysis_agn.bands, fdn_analysis_agn.dfn_amp, ...
              'Color', color_agn, 'LineWidth', 2.5, 'DisplayName', 'Noise Agnostic'); 
    hold on;
    
    h2 = plot(fdn_analysis_awa.bands, fdn_analysis_awa.dfn_amp, ...
              'Color', color_awa, 'LineWidth', 2.5, 'DisplayName', 'Noise Aware'); 
    
    h3 = plot(rir_analysis.bands, rir_analysis.dfn_amp, ...
              'Color', color_gt, 'LineWidth', 2.0, 'LineStyle', '--', ...
              'DisplayName', 'Ground Truth');
   
    ylabel('Amplitude', 'FontSize', 14, 'FontWeight', 'bold')
    xlabel('Frequency [Hz]', 'FontSize', 14, 'FontWeight', 'bold')
    legend('Location', 'best', 'FontSize', 11, 'Box', 'off')
    title('Frequency-dependent Amplitude', 'FontSize', 15, 'FontWeight', 'bold')
    grid on
    set(gca, 'XScale', 'log', 'LineWidth', 1.2, 'Box', 'on')
    xlim([min(rir_analysis.bands), max(rir_analysis.bands)])
    
    % Subplot 2: RT60
    subplot(1, 2, 2)
    plot(fdn_analysis_agn.bands, fdn_analysis_agn.rt60, ...
         'Color', color_agn, 'LineWidth', 2.5, 'DisplayName', 'Noise Agnostic'); 
    hold on;
    
    plot(fdn_analysis_awa.bands, fdn_analysis_awa.rt60, ...
         'Color', color_awa, 'LineWidth', 2.5, 'DisplayName', 'Noise Aware'); 
    
    plot(rir_analysis.bands, rir_analysis.rt60, ...
         'Color', color_gt, 'LineWidth', 2.0, 'LineStyle', '--', ...
         'DisplayName', 'Ground Truth');
    
    ylabel('RT60 [s]', 'FontSize', 14, 'FontWeight', 'bold')
    xlabel('Frequency [Hz]', 'FontSize', 14, 'FontWeight', 'bold')
    legend('Location', 'best', 'FontSize', 11, 'Box', 'off')
    title('Reverberation Time (RT60)', 'FontSize', 15, 'FontWeight', 'bold')
    grid on
    set(gca, 'XScale', 'log', 'LineWidth', 1.2, 'Box', 'on')
    xlim([min(rir_analysis.bands), max(rir_analysis.bands)])
    
    % Add overall title
    sgtitle(rir_name, 'FontSize', 16, 'FontWeight', 'bold', ...
            'Interpreter', 'latex')
    
end

%%

for i_rir=1:n_rirs
    rir_name = strrep(run_dir{rirs_id(i_rir)}, '_', ' ') + " " + loss_dir{i_loss};
    % Create figure with better size and position
    figure('Name', rir_name, 'NumberTitle', 'off', ...
           'Position', [100, 100, 1200, 450], ...
           'Color', 'w');
    
    % Load targets of the rir 
    analysis_file = strrep(run_dir{rirs_id(i_rir)}, 'target_', '') + "_analysis.mat";
    rir_analysis = load(fullfile(rir_filepath, analysis_file));
 
    curr_dir = fullfile(output_folder, test_folder, types(1), run_dir{rirs_id(i_rir)}, loss_dir{i_loss});
    fdn_analysis_agn = load(fullfile(curr_dir, "optimized_rir_analysis.mat"));
    curr_dir = fullfile(output_folder, test_folder, types(2), run_dir{rirs_id(i_rir)}, loss_dir{i_loss});
    fdn_analysis_awa = load(fullfile(curr_dir, "optimized_rir_analysis.mat"));
       
    fs = 48000; %

    edr_agn = fdn_analysis_agn.edr;
    edr_awa = fdn_analysis_awa.edr;
    edr_gt  = rir_analysis.edr;

    % Ensure EDR is [freq x time]
    if size(edr_agn,1) ~= numel(fdn_analysis_agn.bands)
        edr_agn = edr_agn.'; 
    end
    if size(edr_awa,1) ~= numel(fdn_analysis_awa.bands)
        edr_awa = edr_awa.'; 
    end
    if size(edr_gt,1) ~= numel(rir_analysis.bands)
        edr_gt = edr_gt.'; 
    end

    t_fdn = (0:size(edr_awa,2)-1) / fs;
    t_gt  = (0:size(edr_gt,2)-1) / fs;

    f_fdn = (0:size(edr_awa,1)-1) * fs / 2 / size(edr_awa,1);
    f_gt  = (0:size(edr_gt,1)-1) * fs / 2 / size(edr_gt,1);

    ax_1 = subplot(1, 3, 1) 
    % Plot one surface at a time (example: Ground Truth)
    [T, F] = meshgrid(t_gt, f_gt);
    surf(F, T, edr_gt, 'EdgeColor', 'none');
    colormap(ax_1, 'parula'); colorbar
    xlabel('Time [s]', 'FontSize', 14, 'FontWeight', 'bold')
    ylabel('Frequency [Hz]', 'FontSize', 14, 'FontWeight', 'bold')
    zlabel('EDR', 'FontSize', 14, 'FontWeight', 'bold')
    title('EDR (Ground Truth)', 'FontSize', 15, 'FontWeight', 'bold')
    set(gca, 'YScale', 'log', 'LineWidth', 1.2, 'Box', 'on')
    view(135, 30); grid on
    
    ax_2 = subplot(1, 3, 2) 
    % Plot one surface at a time (example: Ground Truth)
    [T, F] = meshgrid(t_fdn, f_fdn); 
    surf(F, T, edr_gt(1:size(edr_agn, 1),:)-edr_agn, 'EdgeColor', 'none');
    colormap(ax_2, cmi); colorbar;clim([-40, 40])
    xlabel('Time [s]', 'FontSize', 14, 'FontWeight', 'bold')
    ylabel('Frequency [Hz]', 'FontSize', 14, 'FontWeight', 'bold')
    zlabel('EDR', 'FontSize', 14, 'FontWeight', 'bold')
    title("Noise Agnostic "+ num2str(mean(abs(edr_gt(1:size(edr_agn, 1),:)-edr_agn), 'all')), 'FontSize', 15, 'FontWeight', 'bold')
    set(gca, 'YScale', 'log', 'LineWidth', 1.2, 'Box', 'on')
    view(135, 30); grid on

    ax_3 = subplot(1, 3, 3) 
    % Plot one surface at a time (example: Ground Truth)
    [T, F] = meshgrid(t_fdn, f_fdn);
    surf(F, T, edr_gt(1:size(edr_awa, 1),:)-edr_awa, 'EdgeColor', 'none');
    colormap(ax_3, cmi); colorbar; clim([-40, 40])
    xlabel('Time [s]', 'FontSize', 14, 'FontWeight', 'bold')
    ylabel('Frequency [Hz]', 'FontSize', 14, 'FontWeight', 'bold')
    zlabel('EDR', 'FontSize', 14, 'FontWeight', 'bold')
    title("Noise Aware " + num2str(mean(abs(edr_gt(1:size(edr_awa, 1),:)-edr_awa), 'all')), 'FontSize', 15, 'FontWeight', 'bold')
    set(gca, 'YScale', 'log', 'LineWidth', 1.2, 'Box', 'on')
    view(135, 30); grid on
end
