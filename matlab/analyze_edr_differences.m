clear all; clc; close all; 

set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',26)

%% DIRs

output_folder = "../shared/output";
ref_rir_folder = "../real_rirs/sdm_selected";
snr_levels = ["snr10_bomb", "snr10_pb132", "snr10_se203", ...
              "snr20_bomb", "snr20_pb132", "snr20_se203"];
loss_types = ["EDC_lin", "EDC_log", "MSS"];
noise_types = ["noise_agnostic", "noise_aware"];

results = struct();

%% PROCESS ALL 
 
rirs_indx = [1 3 4 5 8 9 10 11];

for i_snr = 1:length(snr_levels)
    snr_folder = snr_levels(i_snr);
    snr_path = fullfile(output_folder, snr_folder);
    
    % get SNR value
    snr_parts = split(snr_folder, '_');
    snr_value = snr_parts{1};
    
    fprintf('Processing %s...\n', snr_folder);
    
    % get all result dirs
    dir_info = dir(fullfile(snr_path, 'target_*'));
    target_dirs = {dir_info([dir_info.isdir]).name};
    
    for i_indx = 1:length(rirs_indx)
        i_target = rirs_indx(i_indx);
        target_dir = target_dirs{i_target};
        target_path = fullfile(snr_path, target_dir);
        
        % extract RIR name from target directory
        rir_name = strrep(target_dir, 'target_', '');
        
        fprintf('  Processing RIR: %s\n', rir_name);
        
        ref_analysis_file = fullfile(ref_rir_folder, sprintf('%s_analysis.mat', rir_name));
        target_data = load(ref_analysis_file);
        
        % loop over losses
        for i_loss = 1:length(loss_types)
            loss_type = loss_types(i_loss);
            loss_path = fullfile(target_path, loss_type);
            
            fprintf('    Processing loss: %s\n', loss_type);
            
            % noise agnostic
            noise_agn_pattern = sprintf('%s_noise_agnostic_%s_%s_optimized_rir_analysis.mat', ...
                                        snr_folder, rir_name, loss_type);
            noise_agn_file = fullfile(loss_path, noise_agn_pattern);
            
            % noise aware
            noise_awa_pattern = sprintf('%s_noise_aware_%s_%s_optimized_rir_analysis.mat', ...
                                        snr_folder, rir_name, loss_type);
            noise_awa_file = fullfile(loss_path, noise_awa_pattern);
            
            % Create result entry key
            result_key = sprintf('%s_%s_%s', snr_value, rir_name, loss_type);
            
            % Initialize result structure
            results.(result_key) = struct();
            results.(result_key).snr_level = snr_value;
            results.(result_key).snr_folder = snr_folder;
            results.(result_key).rir_name = rir_name;
            results.(result_key).loss_type = loss_type;
            results.(result_key).target_edr = target_data.edr;
            
            % Process noise agnostic
            agn_data = load(noise_agn_file);
            results.(result_key).agn_edr = agn_data.edr;
            min_dim1 = min(size(target_data.edr, 1), size(agn_data.edr, 1));
            min_dim2 = min(size(target_data.edr, 2), size(agn_data.edr, 2));
            edr_diff_agn = target_data.edr(1:min_dim1, 1:min_dim2) - agn_data.edr(1:min_dim1, 1:min_dim2);
            
            results.(result_key).edr_diff_agn = edr_diff_agn;
            results.(result_key).mean_abs_diff_agn = mean(abs(edr_diff_agn), 'all');
            
            fprintf('      Noise agnostic mean abs diff: %.3f dB\n', results.(result_key).mean_abs_diff_agn);
            
            % Process noise aware
            awa_data = load(noise_awa_file);
            results.(result_key).awa_edr = awa_data.edr;
            min_dim1 = min(size(target_data.edr, 1), size(awa_data.edr, 1));
            min_dim2 = min(size(target_data.edr, 2), size(awa_data.edr, 2));
            edr_diff_awa = target_data.edr(1:min_dim1, 1:min_dim2) - awa_data.edr(1:min_dim1, 1:min_dim2);
            
            results.(result_key).edr_diff_awa = edr_diff_awa;
            results.(result_key).mean_abs_diff_awa = mean(abs(edr_diff_awa), 'all');
            
            fprintf('      Noise aware mean abs diff: %.3f dB\n', results.(result_key).mean_abs_diff_awa);
        end
    end
end

%% SAVE RESULTS

save('edr_analysis_results.mat', 'results');
fprintf('\nResults saved to edr_analysis_results.mat\n');

%% CREATE SUMMARY TABLE

fprintf('\n=== SUMMARY OF EDR DIFFERENCES ===\n\n');

% Extract all result keys
result_keys = fieldnames(results);

% Create summary data
summary_data = [];
row_labels = {};

for i = 1:length(result_keys)
    res = results.(result_keys{i});
    row_label = sprintf('%s | %s | %s', res.snr_level, res.rir_name, res.loss_type);
    row_labels{end+1} = row_label;
    agn_diff = res.mean_abs_diff_agn;
    awa_diff = res.mean_abs_diff_awa;
    summary_data = [summary_data; agn_diff, awa_diff];
end

fprintf('%-60s | %15s | %15s\n', 'Configuration', 'Noise Agnostic', 'Noise Aware');
fprintf('%s\n', repmat('-', 1, 95));

for i = 1:length(row_labels)
    fprintf('%-60s | %15.3f | %15.3f\n', row_labels{i}, summary_data(i, 1), summary_data(i, 2));
end

%% ANALYZE BY SNR AND LOSS TYPE

fprintf('\n=== EDR DIFFERENCES PER LOSS TYPE PER SNR LEVEL ===\n\n');

% Get unique SNR values
unique_snr_vals = unique(cellfun(@(x) extract_snr_number(x), snr_levels, 'UniformOutput', false));

for i_snr_val = 1:length(unique_snr_vals)
    snr_num = unique_snr_vals{i_snr_val};
    fprintf('\n========== SNR LEVEL: %s ==========\n\n', snr_num);
    
    % Create table for this SNR level
    fprintf('%-30s | %18s | %18s\n', 'Loss Type / RIR Name', 'Noise Agnostic (dB)', 'Noise Aware (dB)');
    fprintf('%s\n', repmat('-', 1, 72));
    
    % Process each loss type
    for i_loss = 1:length(loss_types)
        loss_type = loss_types(i_loss);
        fprintf('\n%s:\n', loss_type);
        agn_diffs = []; 
        awa_diffs = []; 
        for i = 1:length(result_keys)
            res = results.(result_keys{i});
            if contains(res.snr_level, snr_num) && strcmp(res.loss_type, loss_type)
                agn_val = res.mean_abs_diff_agn;
                awa_val = res.mean_abs_diff_awa;
                % fprintf('  %-26s | %18.3f | %18.3f\n', res.rir_name, agn_val, awa_val);
                agn_diffs = [agn_diffs; res.mean_abs_diff_agn];
                awa_diffs = [awa_diffs; res.mean_abs_diff_awa];
            end
        end        
        agn_mean = median(agn_diffs);
        awa_mean = median(awa_diffs);
        agn_std = std(agn_diffs);
        awa_std = std(awa_diffs);
        fprintf('%-30s | %8.3f ± %7.3f | %8.3f ± %7.3f\n', ...
                loss_type, agn_mean, agn_std, awa_mean, awa_std);
    end
end

fprintf('\nAnalysis complete!\n');

%% REFERENCE RIR RT60 STATISTICS

fprintf('\n=== REVERBERATION TIME STATISTICS (Reference RIRs from real_rirs/sdm_selected) ===\n\n');

% Load all reference RIR analysis files
ref_dirinfo = dir(fullfile(ref_rir_folder, '*_analysis.mat'));
ref_files = {ref_dirinfo.name};
n_ref_rirs = length(rirs_indx);

all_rt60 = [];
all_bands = [];
ref_rir_names = {};

% load all reference RIR data
for i = 1:n_ref_rirs
    i_rir = rirs_indx(i);
    ref_file = fullfile(ref_rir_folder, ref_files{i});
    ref_data = load(ref_file);
    
    % extract RIR name (remove "_analysis.mat")
    ref_name = strrep(ref_files{i}, '_analysis.mat', '');
    ref_rir_names{i_rir} = ref_name;
    
    all_rt60 = [all_rt60; ref_data.rt60];
    all_bands = ref_data.bands;
end

% Compute statistics
mean_rt60 = mean(all_rt60, 1);
std_rt60 = std(all_rt60, 1);
min_rt60 = min(mean(all_rt60, 2), [], 1);
max_rt60 = max(mean(all_rt60, 2), [], 1);

% Display overall statistics
fprintf('Number of reference RIRs: %d\n', n_ref_rirs);
fprintf('Frequency bands: %d\n\n', length(all_bands));

fprintf('Overall RT60 Statistics:\n');
fprintf('  Mean: %.3f s\n', mean(mean_rt60));
fprintf('  Std Dev: %.3f s\n', std(mean_rt60));
fprintf('  Min: %.3f s\n',min_rt60);
fprintf('  Max: %.3f s\n\n', max_rt60);

%% HELPER FUNCTION

function snr_num = extract_snr_number(snr_str)
    % Extract SNR number 
    parts = split(snr_str, '_');
    snr_num = parts{1};
end

