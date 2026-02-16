clear all; clc; close all; 

set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',26)

% include yaml library to read the configuration file 
addpath("+yaml")
addpath("./Colormaps/")
mycolormap = customcolormap([0 .25 .5 .75 1], {'#9d0142','#f66e45','#ffffbb','#65c0ae','#5e4f9f'});

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
rirs_id  = [3]; % [1:4]; % 
n_rirs = length(rirs_id);

fullfilepath = fullfile(output_folder, test_folder, types(1), run_dir{1});
dirinfo = dir(fullfilepath);
dirinfo = dirinfo([dirinfo.isdir]);
dirnames = {dirinfo.name};
loss_dir = dirnames(~ismember(dirnames, {'.', '..'}));

i_loss = 3; 
rir_filepath = fullfile("..", "real_rirs", "sdm_selected");

%% Define colors
color_agn = [0.2, 0.4, 0.8];      % Blue
color_awa = [0.9, 0.3, 0.2];      % Red
color_gt = [0.2, 0.2, 0.2];       % Dark gray

%%

% for i_rir=1:n_rirs
%     rir_name = strrep(run_dir{rirs_id(i_rir)}, '_', ' ') + " " + loss_dir{i_loss};
%     
%     % Create figure with better size and position
%     figure('Name', rir_name, 'NumberTitle', 'off', ...
%            'Position', [100, 100, 1200, 450], ...
%            'Color', 'w');
%     
%     % Create tiled layout
%     tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
%     
%     % Load targets of the rir 
%     analysis_file = strrep(run_dir{rirs_id(i_rir)}, 'target_', '') + "_analysis.mat";
%     rir_analysis = load(fullfile(rir_filepath, analysis_file));
%  
%     curr_dir = fullfile(output_folder, test_folder, types(1), run_dir{rirs_id(i_rir)}, loss_dir{i_loss});
%     fdn_analysis_agn = load(fullfile(curr_dir, "optimized_rir_analysis.mat"));
%     curr_dir = fullfile(output_folder, test_folder, types(2), run_dir{rirs_id(i_rir)}, loss_dir{i_loss});
%     fdn_analysis_awa = load(fullfile(curr_dir, "optimized_rir_analysis.mat"));
% 
%     % Tile 1: Amplitude
%     nexttile    
%     h1 = plot(fdn_analysis_agn.bands, fdn_analysis_agn.dfn_amp, ...
%               'Color', color_agn, 'LineWidth', 2.5, 'DisplayName', 'Noise Agnostic'); 
%     hold on;
%     
%     h2 = plot(fdn_analysis_awa.bands, fdn_analysis_awa.dfn_amp, ...
%               'Color', color_awa, 'LineWidth', 2.5, 'DisplayName', 'Noise Aware'); 
%     
%     h3 = plot(rir_analysis.bands, rir_analysis.dfn_amp, ...
%               'Color', color_gt, 'LineWidth', 2.0, 'LineStyle', '--', ...
%               'DisplayName', 'Ground Truth');
%    
%     ylabel('Amplitude', 'FontSize', 30, 'Interpreter','latex')
%     xlabel('Frequency (kHz)', 'FontSize', 30, 'Interpreter','latex')
%     legend('Location', 'best', 'FontSize', 11, 'Box', 'off')
%     title('Frequency-dependent Amplitude', 'FontSize', 15, 'Interpreter','latex')
%     
%     set(gca, 'XScale', 'log', 'LineWidth', 1.2, 'Box', 'on')
%     xlim([min(rir_analysis.bands), max(rir_analysis.bands)])
%     
%     % Tile 2: RT60
%     nexttile
%     plot(fdn_analysis_agn.bands, fdn_analysis_agn.rt60, ...
%          'Color', color_agn, 'LineWidth', 2.5, 'DisplayName', 'Noise Agnostic'); 
%     hold on;
%     
%     plot(fdn_analysis_awa.bands, fdn_analysis_awa.rt60, ...
%          'Color', color_awa, 'LineWidth', 2.5, 'DisplayName', 'Noise Aware'); 
%     
%     plot(rir_analysis.bands, rir_analysis.rt60, ...
%          'Color', color_gt, 'LineWidth', 2.0, 'LineStyle', '--', ...
%          'DisplayName', 'Ground Truth');
%     
%     ylabel('RT60 (s)', 'FontSize', 30, 'Interpreter','latex')
%     xlabel('Frequency (kHz)', 'FontSize', 30, 'Interpreter','latex')
%     legend('Location', 'best', 'FontSize', 11, 'Box', 'off')
%     title('Reverberation Time (RT60)', 'FontSize', 15, 'Interpreter','latex')
%     
%     set(gca, 'XScale', 'log', 'LineWidth', 1.2, 'Box', 'on')
%     xlim([min(rir_analysis.bands), max(rir_analysis.bands)])
%     
%     % Add overall title
%     sgtitle(rir_name, 'FontSize', 16, 'Interpreter','latex', ...
%             'Interpreter', 'latex')
%     
% end

%%

for i_rir=1:n_rirs
    % Load targets of the rir 
    analysis_file = strrep(run_dir{rirs_id(i_rir)}, 'target_', '') + "_analysis.mat";
    rir_analysis = load(fullfile(rir_filepath, analysis_file));
    
    fs = 48000;
    
    edr_gt  = rir_analysis.edr;
    
    % Ensure EDR is [freq x time]
    if size(edr_gt,1) ~= numel(rir_analysis.bands)
        edr_gt = edr_gt.'; 
    end
    
    t_gt  = linspace(0, 2, size(edr_gt,1));
    f_gt  = (0:size(edr_gt,2)-1) * fs / 2 / size(edr_gt,2);
    
    % FIGURE 1: Reference EDR only
    rir_name = strrep(run_dir{rirs_id(i_rir)}, '_', ' ');
    figure('Name', [rir_name ' - Reference'], 'NumberTitle', 'off', ...
           'Position', [60 10 700 350], ...
           'Color', 'w');
    
    ax_ref = gca;
    imagesc(t_gt, f_gt, edr_gt.');
    set(gca, 'YDir', 'normal');
    colormap(ax_ref, mycolormap); 
    cb_ref = colorbar;
    cb_ref.TickLabelInterpreter = 'latex';
    clim([-100, 20])
    ylabel(cb_ref, 'Magnitude (dB)', 'FontSize', 24, 'Interpreter','latex');
    ylabel('Frequency (kHz)', 'FontSize', 30, 'Interpreter','latex')
    xlabel('Time (s)', 'FontSize', 30, 'Interpreter','latex')    
    yticks(ax_ref, [1000, 5000, 10000, 15000, 20000]);    
    yticklabels(ax_ref, {'1', '5', '10', '15', '20'});
    title('$h(t) + w_1(t)$', 'FontSize', 30, 'Interpreter','latex')
    set(gca, 'LineWidth', 1.2, 'Box', 'on')
    
    xlim([0, 1.5])
    
    % FIGURES 2-4: One per loss function
    for i_loss = 1:length(loss_dir)
        loss_name = loss_dir{i_loss};
        
        % Load FDN analysis data
        curr_dir = fullfile(output_folder, test_folder, types(1), run_dir{rirs_id(i_rir)}, loss_name);
        fdn_analysis_agn = load(fullfile(curr_dir, "optimized_rir_analysis.mat"));
        curr_dir = fullfile(output_folder, test_folder, types(2), run_dir{rirs_id(i_rir)}, loss_name);
        fdn_analysis_awa = load(fullfile(curr_dir, "optimized_rir_analysis.mat"));
        
        edr_agn = fdn_analysis_agn.edr;
        edr_awa = fdn_analysis_awa.edr;
        
        % Ensure EDR is [freq x time]
        if size(edr_agn,1) ~= numel(fdn_analysis_agn.bands)
            edr_agn = edr_agn.'; 
        end
        if size(edr_awa,1) ~= numel(fdn_analysis_awa.bands)
            edr_awa = edr_awa.'; 
        end
        
        t_fdn = linspace(0, 2, size(edr_awa,1));
        f_fdn = (0:size(edr_awa,2)-1) * fs / 2 / size(edr_awa,2);
        
        % Create figure for this loss function
        fig_name = [rir_name ' - ' loss_name];
        figure('Name', fig_name, 'NumberTitle', 'off', ...
               'Position', [60 10 1000 350], ...
               'Color', 'w');
        
        % Create tiled layout
        tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
        
        % Tile 1: Noise Agnostic difference
        ax1 = nexttile;
        imagesc(t_fdn, f_fdn, (edr_gt(1:size(edr_agn, 1),:)-edr_agn).');
        set(gca, 'YDir', 'normal');
        colormap(ax1, cmi); 
%         cb1 = colorbar;
%         cb1.TickLabelInterpreter = 'latex';
%         ylabel(cb1, 'Magnitude (dB)', 'FontSize', 24, 'Interpreter','latex');
        clim([-40, 40])
        ylabel('Frequency (kHz)', 'FontSize', 30, 'Interpreter','latex')
        xlabel('Time (s)', 'FontSize', 30, 'Interpreter','latex')
        yticks(ax1, [1000, 5000, 10000, 15000, 20000]);
        yticklabels(ax1, {'1', '5', '10', '15', '20'});
        title(num2str(mean(abs(edr_gt(1:size(edr_agn, 1),:)-edr_agn), 'all'), 3), 'FontSize', 30, 'Interpreter','latex')
        set(gca, 'LineWidth', 1.2, 'Box', 'on')
        
        xlim([0, 1.5])
        
        % Tile 2: Noise Aware difference
        ax2 = nexttile;
        imagesc(t_fdn, f_fdn, (edr_gt(1:size(edr_awa, 1),:)-edr_awa).');
        set(gca, 'YDir', 'normal');
        colormap(ax2, cmi); 
        cb2 = colorbar; 
        cb2.TickLabelInterpreter = 'latex';
        ylabel(cb2, 'Magnitude (dB)', 'FontSize', 24, 'Interpreter','latex');
        clim([-40, 40])
        % ylabel('Frequency (kHz)', 'FontSize', 30, 'Interpreter','latex')
        xlabel('Time (s)', 'FontSize', 30, 'Interpreter','latex')
        yticks(ax2, [1000, 5000, 10000, 15000, 20000]);
        yticklabels(ax2, {'1', '5', '10', '15', '20'});
        title(num2str(mean(abs(edr_gt(1:size(edr_awa, 1),:)-edr_awa), 'all'), 3), 'FontSize', 30, 'Interpreter','latex')
        set(gca, 'LineWidth', 1.2, 'Box', 'on')
        
        xlim([0, 1.5])
        
        % Export each loss function figure
        exportgraphics(gcf, sprintf('edr_diff_%s.eps', loss_name), 'ContentType', 'vector', 'Resolution', 100);
    end
    
end
