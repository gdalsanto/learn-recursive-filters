clear all; clc; close all; 

set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',20)

% include yaml library to read the configuration file 
addpath("+yaml")

%% LOAD DATA I

output_folder = "../output";
test_folder = "snr10_bomb"; 
types = ["noise_agnostic", "noise_aware"];
loss_id = [1, 2, 3];
yml_file = "optimization_real_data_gpu.yml";

fullfilepath = fullfile(output_folder, test_folder, types(1));
dirinfo = dir(fullfilepath);
dirinfo = dirinfo([dirinfo.isdir]);
dirnames = {dirinfo.name};
run_dir = dirnames(~ismember(dirnames, {'.', '..'}));
n_rirs = length(run_dir);

fullfilepath = fullfile(output_folder, test_folder, types(1), run_dir{1});
dirinfo = dir(fullfilepath);
dirinfo = dirinfo([dirinfo.isdir]);
dirnames = {dirinfo.name};
loss_dir = dirnames(~ismember(dirnames, {'.', '..'}));

i_loss = 1; 
rir_filepath = fullfile("..", "real_rirs", "sdm_selected");

%%

% tc_fbans = [0, 44.1942, 62.5000, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 22627.416];
for i_rir=1:n_rirs
    rir_name = strrep(run_dir{i_rir}, '_', ' ') + " " + loss_dir{i_loss};
    figure('Name', rir_name, 'NumberTitle', 'off');
    % laod targets of the rir 
    analysis_file = strrep(run_dir{i_rir}, 'target_', '') + "_analysis.mat";
    rir_analysis = load(fullfile(rir_filepath, analysis_file));
    i_plot = 1;
 
    curr_dir = fullfile(output_folder, test_folder, types(1), run_dir{i_rir}, loss_dir{i_loss});
    fdn_analysis_agn = load(fullfile(curr_dir, "optimized_rir_analysis.mat"));
    curr_dir = fullfile(output_folder, test_folder, types(2), run_dir{i_rir}, loss_dir{i_loss});
    fdn_analysis_awa = load(fullfile(curr_dir, "optimized_rir_analysis.mat"));


    subplot(1, 2, 1)    
    plot(fdn_analysis.bands, fdn_analysis_agn.dfn_amp, 'b-'); hold on 

    plot(fdn_analysis.bands, fdn_analysis_awa.dfn_amp, 'r-'); hold on 
    plot(rir_analysis.bands, rir_analysis.dfn_amp, 'k-');
   
    ylabel('A')
    xlabel('Freq')
    legend('AGN', "AW", 'GT')
    title(types(i_type))
    grid on 
    
    subplot(1, 2, 2)
    plot(fdn_analysis.bands, fdn_analysis_agn.rt60, 'b-'); hold on 
    plot(fdn_analysis.bands, fdn_analysis_awa.rt60, 'r-'); hold on 
    plot(rir_analysis.bands, rir_analysis.rt60, 'k-');
    ylabel('RT60')
    xlabel('Freq')
    legend('AGN', "AW", 'GT')
    title(types(i_type))
    grid on 

    i_plot = i_plot + 1 ;
    
end