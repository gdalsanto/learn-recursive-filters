clear all; clc; close all; 

set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',26)

%% CONFIGURATION

% Base directory for the output data
output_folder = "../shared/output";

% Reference RIR analysis directory
ref_rir_folder = "../real_rirs/sdm_selected";

% SNR levels to process
snr_levels = ["snr10_bomb", "snr10_pb132", "snr10_se203", ...
              "snr20_bomb", "snr20_pb132", "snr20_se203"];

% Loss function types
loss_types = ["EDC_lin", "EDC_log", "MSS"];

% Noise types
noise_types = ["noise_agnostic", "noise_aware"];

%% INITIALIZE RESULTS STRUCTURE

results = struct();

%% PROCESS ALL SNR LEVELS AND LOSS FUNCTIONS

for i_snr = 1:length(snr_levels)
    snr_folder = snr_levels(i_snr);
    snr_path = fullfile(output_folder, snr_folder);
    
    % Extract SNR value from folder name (e.g., "snr10_bomb" -> "snr10")
    snr_parts = split(snr_folder, '_');
    snr_value = snr_parts{1};
    
    fprintf('Processing %s...\n', snr_folder);
    
    % Check if SNR folder exists
    if ~exist(snr_path, 'dir')
        warning('SNR folder does not exist: %s', snr_path);
        continue;
    end
    
    % Get all target RIR directories (e.g., target_SDMdata_CH_BB)
    dir_info = dir(fullfile(snr_path, 'target_*'));
    target_dirs = {dir_info([dir_info.isdir]).name};
    
    for i_target = 1:length(target_dirs)
        target_dir = target_dirs{i_target};
        target_path = fullfile(snr_path, target_dir);
        
        % Extract RIR name from target directory (e.g., "target_SDMdata_CH_BB" -> "SDMdata_CH_BB")
        rir_name = strrep(target_dir, 'target_', '');
        
        fprintf('  Processing RIR: %s\n', rir_name);
        
        % Load target RIR analysis from real_rirs/sdm_selected
        % The reference RIR files have names like "SDMdata_CH_BB_analysis.mat"
        ref_analysis_file = fullfile(ref_rir_folder, sprintf('%s_analysis.mat', rir_name));
        if ~exist(ref_analysis_file, 'file')
            warning('    Reference RIR analysis file not found: %s', ref_analysis_file);
            continue;
        end
        target_data = load(ref_analysis_file);
        
        % Process each loss function type
        for i_loss = 1:length(loss_types)
            loss_type = loss_types(i_loss);
            loss_path = fullfile(target_path, loss_type);
            
            % Check if loss folder exists
            if ~exist(loss_path, 'dir')
                continue;
            end
            
            fprintf('    Processing loss: %s\n', loss_type);
            
            % Process noise agnostic
            noise_agn_pattern = sprintf('%s_noise_agnostic_%s_%s_optimized_rir_analysis.mat', ...
                                        snr_folder, rir_name, loss_type);
            noise_agn_file = fullfile(loss_path, noise_agn_pattern);
            
            % Process noise aware
            noise_awa_pattern = sprintf('%s_noise_aware_%s_%s_optimized_rir_analysis.mat', ...
                                        snr_folder, rir_name, loss_type);
            noise_awa_file = fullfile(loss_path, noise_awa_pattern);
            
            % Check if files exist
            has_agn = exist(noise_agn_file, 'file');
            has_awa = exist(noise_awa_file, 'file');
            
            if ~has_agn && ~has_awa
                warning('      No analysis files found for loss type: %s', loss_type);
                continue;
            end
            
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
            if has_agn
                agn_data = load(noise_agn_file);
                results.(result_key).agn_edr = agn_data.edr;
                
                % Compute EDR difference
                % Ensure dimensions match (crop to minimum size)
                min_dim1 = min(size(target_data.edr, 1), size(agn_data.edr, 1));
                min_dim2 = min(size(target_data.edr, 2), size(agn_data.edr, 2));
                edr_diff_agn = target_data.edr(1:min_dim1, 1:min_dim2) - agn_data.edr(1:min_dim1, 1:min_dim2);
                
                results.(result_key).edr_diff_agn = edr_diff_agn;
                results.(result_key).mean_abs_diff_agn = mean(abs(edr_diff_agn), 'all');
                
                fprintf('      Noise agnostic mean abs diff: %.3f dB\n', results.(result_key).mean_abs_diff_agn);
            end
            
            % Process noise aware
            if has_awa
                awa_data = load(noise_awa_file);
                results.(result_key).awa_edr = awa_data.edr;
                
                % Compute EDR difference
                min_dim1 = min(size(target_data.edr, 1), size(awa_data.edr, 1));
                min_dim2 = min(size(target_data.edr, 2), size(awa_data.edr, 2));
                edr_diff_awa = target_data.edr(1:min_dim1, 1:min_dim2) - awa_data.edr(1:min_dim1, 1:min_dim2);
                
                results.(result_key).edr_diff_awa = edr_diff_awa;
                results.(result_key).mean_abs_diff_awa = mean(abs(edr_diff_awa), 'all');
                
                fprintf('      Noise aware mean abs diff: %.3f dB\n', results.(result_key).mean_abs_diff_awa);
            end
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
    key = result_keys{i};
    res = results.(key);
    
    row_label = sprintf('%s | %s | %s', res.snr_level, res.rir_name, res.loss_type);
    row_labels{end+1} = row_label;
    
    % Get mean absolute differences
    agn_diff = NaN;
    awa_diff = NaN;
    
    if isfield(res, 'mean_abs_diff_agn')
        agn_diff = res.mean_abs_diff_agn;
    end
    
    if isfield(res, 'mean_abs_diff_awa')
        awa_diff = res.mean_abs_diff_awa;
    end
    
    summary_data = [summary_data; agn_diff, awa_diff];
end

% Display summary table
fprintf('%-60s | %15s | %15s\n', 'Configuration', 'Noise Agnostic', 'Noise Aware');
fprintf('%s\n', repmat('-', 1, 95));

for i = 1:length(row_labels)
    fprintf('%-60s | %15.3f | %15.3f\n', row_labels{i}, summary_data(i, 1), summary_data(i, 2));
end

%% ANALYZE BY SNR LEVEL

fprintf('\n=== MEAN DIFFERENCES BY SNR LEVEL ===\n\n');

% Extract unique SNR levels
snr_values = {};
for i = 1:length(result_keys)
    snr_values{end+1} = results.(result_keys{i}).snr_level;
end
unique_snr = unique(snr_values);

for i_snr = 1:length(unique_snr)
    snr_str = unique_snr{i_snr};
    
    % Find all results for this SNR level
    agn_diffs = [];
    awa_diffs = [];
    
    for i = 1:length(result_keys)
        key = result_keys{i};
        if strcmp(results.(key).snr_level, snr_str)
            if isfield(results.(key), 'mean_abs_diff_agn')
                agn_diffs = [agn_diffs; results.(key).mean_abs_diff_agn];
            end
            if isfield(results.(key), 'mean_abs_diff_awa')
                awa_diffs = [awa_diffs; results.(key).mean_abs_diff_awa];
            end
        end
    end
    
    fprintf('%s:\n', upper(snr_str));
    fprintf('  Noise Agnostic: %.3f ± %.3f dB (n=%d)\n', ...
            mean(agn_diffs), std(agn_diffs), length(agn_diffs));
    fprintf('  Noise Aware:    %.3f ± %.3f dB (n=%d)\n\n', ...
            mean(awa_diffs), std(awa_diffs), length(awa_diffs));
end

%% ANALYZE BY LOSS TYPE

fprintf('\n=== MEAN DIFFERENCES BY LOSS TYPE ===\n\n');

for loss_val = loss_types
    loss_str = loss_val;
    
    % Find all results for this loss type
    agn_diffs = [];
    awa_diffs = [];
    
    for i = 1:length(result_keys)
        key = result_keys{i};
        if strcmp(results.(key).loss_type, loss_str)
            if isfield(results.(key), 'mean_abs_diff_agn')
                agn_diffs = [agn_diffs; results.(key).mean_abs_diff_agn];
            end
            if isfield(results.(key), 'mean_abs_diff_awa')
                awa_diffs = [awa_diffs; results.(key).mean_abs_diff_awa];
            end
        end
    end
    
    fprintf('%s:\n', loss_str);
    fprintf('  Noise Agnostic: %.3f ± %.3f dB (n=%d)\n', ...
            mean(agn_diffs), std(agn_diffs), length(agn_diffs));
    fprintf('  Noise Aware:    %.3f ± %.3f dB (n=%d)\n\n', ...
            mean(awa_diffs), std(awa_diffs), length(awa_diffs));
end

%% ANALYZE BY SNR AND LOSS TYPE - CREATE DETAILED TABLES

fprintf('\n=== DETAILED ANALYSIS: EDR DIFFERENCES PER LOSS TYPE PER SNR LEVEL ===\n\n');

% Get unique SNR values
unique_snr_vals = unique(cellfun(@(x) extract_snr_number(x), unique_snr, 'UniformOutput', false));

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
        
        % Find all results for this SNR and loss type
        for i = 1:length(result_keys)
            key = result_keys{i};
            res = results.(key);
            
            % Check if this result matches the SNR and loss type
            if contains(res.snr_level, snr_num) && strcmp(res.loss_type, loss_type)
                agn_val = NaN;
                awa_val = NaN;
                
                if isfield(res, 'mean_abs_diff_agn')
                    agn_val = res.mean_abs_diff_agn;
                end
                
                if isfield(res, 'mean_abs_diff_awa')
                    awa_val = res.mean_abs_diff_awa;
                end
                
                fprintf('  %-26s | %18.3f | %18.3f\n', res.rir_name, agn_val, awa_val);
            end
        end
    end
    
    % Compute and display averages for this SNR level
    fprintf('\n%s\n', repmat('-', 1, 72));
    fprintf('%-30s | %18s | %18s\n', 'AVERAGE', 'Noise Agnostic (dB)', 'Noise Aware (dB)');
    fprintf('%s\n', repmat('-', 1, 72));
    
    for i_loss = 1:length(loss_types)
        loss_type = loss_types(i_loss);
        
        agn_diffs = [];
        awa_diffs = [];
        
        for i = 1:length(result_keys)
            key = result_keys{i};
            res = results.(key);
            
            if contains(res.snr_level, snr_num) && strcmp(res.loss_type, loss_type)
                if isfield(res, 'mean_abs_diff_agn')
                    agn_diffs = [agn_diffs; res.mean_abs_diff_agn];
                end
                if isfield(res, 'mean_abs_diff_awa')
                    awa_diffs = [awa_diffs; res.mean_abs_diff_awa];
                end
            end
        end
        
        if ~isempty(agn_diffs) || ~isempty(awa_diffs)
            agn_mean = NaN;
            awa_mean = NaN;
            
            if ~isempty(agn_diffs)
                agn_mean = median(agn_diffs);
            end
            if ~isempty(awa_diffs)
                awa_mean = median(awa_diffs);
            end
            
            fprintf('  %-26s | %18.3f | %18.3f\n', loss_type, agn_mean, awa_mean);
        end
    end
end

fprintf('\nAnalysis complete!\n');

%% REFERENCE RIR RT60 STATISTICS

fprintf('\n=== REVERBERATION TIME STATISTICS (Reference RIRs from real_rirs/sdm_selected) ===\n\n');

% Load all reference RIR analysis files
ref_dirinfo = dir(fullfile(ref_rir_folder, '*_analysis.mat'));
ref_files = {ref_dirinfo.name};
n_ref_rirs = length(ref_files);

if n_ref_rirs == 0
    warning('No reference RIR analysis files found in %s', ref_rir_folder);
else
    % Initialize storage for RT60 data
    all_rt60 = [];
    all_bands = [];
    ref_rir_names = {};
    
    % Load all reference RIR data
    for i = 1:n_ref_rirs
        ref_file = fullfile(ref_rir_folder, ref_files{i});
        ref_data = load(ref_file);
        
        % Extract RIR name (remove "_analysis.mat")
        ref_name = strrep(ref_files{i}, '_analysis.mat', '');
        ref_rir_names{i} = ref_name;
        
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
    fprintf('  Min: %.3f s\n', min(min_rt60));
    fprintf('  Max: %.3f s\n\n', max(max_rt60));
    
    % Display per-frequency statistics
    fprintf('Per-Frequency RT60 Statistics:\n');
    fprintf('Frequency (kHz) | Mean RT60 (s) | Std Dev (s) | Min (s) | Max (s)\n');
    fprintf('---------------------------------------------------------------\n');
    for i = 1:length(all_bands)
        fprintf('%13.2f  |    %9.4f   |   %9.4f  | %7.4f | %7.4f\n', ...
                all_bands(i), mean_rt60(i), std_rt60(i), min_rt60(i), max_rt60(i));
    end
    
    % Display individual RIR names
    fprintf('\n\nReference RIRs analyzed:\n');
    for i = 1:n_ref_rirs
        fprintf('%2d. %s\n', i, ref_rir_names{i});
    end
end

%% HELPER FUNCTION

function snr_num = extract_snr_number(snr_str)
    % Extract SNR number (e.g., "snr10" -> "snr10")
    parts = split(snr_str, '_');
    snr_num = parts{1};
end

