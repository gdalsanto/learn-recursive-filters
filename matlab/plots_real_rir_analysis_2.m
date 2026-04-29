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

output_folder = "../shared";
reference_folder = fullfile("..", "real_rirs", "sdm_selected");
website_audio_folder = fullfile("..", "website", "audio");
spectrogram_folder = fullfile(website_audio_folder, "spectrograms");
types = ["noise_agnostic", "noise_aware"];
yml_file = "optimization_real_data_gpu.yml";

snr_dirinfo = dir(output_folder);
snr_dirinfo = snr_dirinfo([snr_dirinfo.isdir]);
snr_folders = {snr_dirinfo.name};
snr_folders = snr_folders(startsWith(snr_folders, "snr"));

% Store MAE values for later use in website tables.
mae_values = struct();

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

for i_snr = 1:length(snr_folders)
    snr_folder = snr_folders{i_snr};
    snr_path = fullfile(output_folder, snr_folder);
    snr_tokens = split(string(snr_folder), "_");
    if numel(snr_tokens) < 2
        warning('Unexpected SNR folder format: %s', snr_folder);
        continue;
    end
    snr_key = char(snr_tokens(1));
    noise_key = char(join(snr_tokens(2:end), "_"));

    target_info = dir(fullfile(snr_path, types(1), 'target_*'));
    target_info = target_info([target_info.isdir]);
    target_dirs = {target_info.name};
    % remove the 
    for i_target = 1:length(target_dirs)
        target_dir = target_dirs{i_target};
        rir_name = strrep(target_dir, 'target_', '');

        analysis_file = fullfile(reference_folder, rir_name + "_analysis.mat");
        if ~exist(analysis_file, 'file')
            warning('Missing reference analysis file: %s', analysis_file);
            continue;
        end

        rir_analysis = load(analysis_file);

        fs = 48000;

        edr_gt = rir_analysis.edr;

        % Ensure EDR is [freq x time]
        if size(edr_gt, 1) ~= numel(rir_analysis.bands)
            edr_gt = edr_gt.';
        end

        t_gt = linspace(0, 2, size(edr_gt, 1));
        f_gt = (0:size(edr_gt, 2)-1) * fs / 2 / size(edr_gt, 2);

        % FIGURE 1: Reference EDR only
        fig_ref = figure('Name', [snr_folder ' - ' rir_name ' - Reference'], 'NumberTitle', 'off', ...
            'Position', [60 10 700 350], ...
            'Color', 'w');

        ax_ref = gca;
        imagesc(t_gt, f_gt, edr_gt.');
        set(gca, 'YDir', 'normal');
        colormap(ax_ref, mycolormap);
        cb_ref = colorbar;
        cb_ref.TickLabelInterpreter = 'latex';
        clim([-100, 20])
        ylabel(cb_ref, 'Magnitude (dB)', 'FontSize', 24, 'Interpreter', 'latex');
        ylabel('Frequency (kHz)', 'FontSize', 30, 'Interpreter', 'latex')
        xlabel('Time (s)', 'FontSize', 30, 'Interpreter', 'latex')
        yticks(ax_ref, [1000, 5000, 10000, 15000, 20000]);
        yticklabels(ax_ref, {'1', '5', '10', '15', '20'});
        % title('$h(t)$', 'FontSize', 30, 'Interpreter', 'latex')
        set(gca, 'LineWidth', 1.2, 'Box', 'on')

        xlim([0, 1.5])

        ref_wav_path = fullfile("..", "website", "audio", "sdm_selected", rir_name + ".wav");
        export_edr_plot(fig_ref, ref_wav_path, spectrogram_folder);
        close(fig_ref);

        loss_info = dir(fullfile(snr_path, types(1), target_dir));
        loss_info = loss_info([loss_info.isdir]);
        loss_dirs = {loss_info.name};
        loss_dirs = loss_dirs(~ismember(loss_dirs, {'.', '..'}));

        % FIGURES: one per loss function and per noise condition
        for i_loss = 1:length(loss_dirs)
            loss_name = loss_dirs{i_loss};

                agn_dir = fullfile(snr_path, types(1), target_dir, loss_name);
                awa_dir = fullfile(snr_path, types(2), target_dir, loss_name);
                agn_file = fullfile(agn_dir, "optimized_rir_analysis.mat");
                awa_file = fullfile(awa_dir, "optimized_rir_analysis.mat");

                fdn_analysis_agn = load(agn_file);
                fdn_analysis_awa = load(awa_file);

            edr_agn = fdn_analysis_agn.edr;
            edr_awa = fdn_analysis_awa.edr;

            % Ensure EDR is [freq x time]
            if size(edr_agn, 1) ~= numel(fdn_analysis_agn.bands)
                edr_agn = edr_agn.';
            end
            if size(edr_awa, 1) ~= numel(fdn_analysis_awa.bands)
                edr_awa = edr_awa.';
            end

            t_fdn = linspace(0, 2, size(edr_awa, 1));
            f_fdn = (0:size(edr_awa, 2)-1) * fs / 2 / size(edr_awa, 2);
            min_time = min(size(edr_gt, 1), size(edr_agn, 1));
                 diff_agn = edr_gt(1:min_time, :) - edr_agn(1:min_time, :);
                 mae_agn = mean(abs(diff_agn), 'all');
            % Noise agnostic difference figure
            fig_agn = figure('Name', [snr_folder ' - ' rir_name ' - ' loss_name ' - Noise Agnostic'], ...
                   'NumberTitle', 'off', ...
                   'Position', [60 10 700 350], ...
                   'Color', 'w');
            ax_agn = gca;
                 imagesc(t_fdn, f_fdn, diff_agn.');
            set(gca, 'YDir', 'normal');
            colormap(ax_agn, cmi);
            cb_agn = colorbar;
            cb_agn.TickLabelInterpreter = 'latex';
            ylabel(cb_agn, 'Magnitude (dB)', 'FontSize', 24, 'Interpreter', 'latex');
            clim([-40, 40])
            ylabel('Frequency (kHz)', 'FontSize', 30, 'Interpreter', 'latex')
            xlabel('Time (s)', 'FontSize', 30, 'Interpreter', 'latex')
            yticks(ax_agn, [1000, 5000, 10000, 15000, 20000]);
            yticklabels(ax_agn, {'1', '5', '10', '15', '20'});
            title(sprintf('%.3f dB', mae_agn), ...
                  'FontSize', 26, 'Interpreter', 'latex')
            set(gca, 'LineWidth', 1.2, 'Box', 'on')
            xlim([0, 1.5])

                agn_wav_path = fullfile("..", "website", "audio", snr_folder, target_dir, loss_name, ...
                                        sprintf('%s_noise_agnostic_%s_%s_optimized_rir.wav', ...
                                        snr_folder, rir_name, loss_name));
                export_edr_plot(fig_agn, agn_wav_path, spectrogram_folder);
                close(fig_agn);

            % Noise aware difference figure
            min_time = min(size(edr_gt, 1), size(edr_awa, 1));
                 diff_awa = edr_gt(1:min_time, :) - edr_awa(1:min_time, :);
                 mae_awa = mean(abs(diff_awa), 'all');
            fig_awa = figure('Name', [snr_folder ' - ' rir_name ' - ' loss_name ' - Noise Aware'], ...
                   'NumberTitle', 'off', ...
                   'Position', [60 10 700 350], ...
                   'Color', 'w');
            ax_awa = gca;
                 imagesc(t_fdn, f_fdn, diff_awa.');
            set(gca, 'YDir', 'normal');
            colormap(ax_awa, cmi);
            cb_awa = colorbar;
            cb_awa.TickLabelInterpreter = 'latex';
            ylabel(cb_awa, 'Magnitude (dB)', 'FontSize', 24, 'Interpreter', 'latex');
            clim([-40, 40])
            ylabel('Frequency (kHz)', 'FontSize', 30, 'Interpreter', 'latex')
            xlabel('Time (s)', 'FontSize', 30, 'Interpreter', 'latex')
            yticks(ax_awa, [1000, 5000, 10000, 15000, 20000]);
            yticklabels(ax_awa, {'1', '5', '10', '15', '20'});
            title(sprintf('%.3f dB', mae_awa), ...
                  'FontSize', 26, 'Interpreter', 'latex')
            set(gca, 'LineWidth', 1.2, 'Box', 'on')
            xlim([0, 1.5])

                awa_wav_path = fullfile("..", "website", "audio", snr_folder, target_dir, loss_name, ...
                                        sprintf('%s_noise_aware_%s_%s_optimized_rir.wav', ...
                                        snr_folder, rir_name, loss_name));
                export_edr_plot(fig_awa, awa_wav_path, spectrogram_folder);
                close(fig_awa);

                mae_values.(snr_key).(noise_key).(rir_name).(loss_name).noise_agnostic = mae_agn;
                mae_values.(snr_key).(noise_key).(rir_name).(loss_name).noise_aware = mae_awa;
        end
    end
end

mae_output_mat = fullfile(website_audio_folder, "mae_values.mat");
mae_output_json = fullfile(website_audio_folder, "mae_values.json");
save(mae_output_mat, 'mae_values');
write_text_file(mae_output_json, jsonencode(mae_values));


function export_edr_plot(fig_handle, wav_path, spectrogram_folder)
    png_path = wav_to_spectrogram_path(wav_path, spectrogram_folder);
    png_dir = fileparts(png_path);
    if ~exist(png_dir, 'dir')
        mkdir(png_dir);
    end
    exportgraphics(fig_handle, png_path, 'Resolution', 200);
end

function png_path = wav_to_spectrogram_path(wav_path, spectrogram_folder)
    wav_path = string(wav_path);
    wav_path = replace(wav_path, "\\", "/");

    if startsWith(wav_path, "./audio/")
        relative_path = extractAfter(wav_path, "./audio/");
    elseif startsWith(wav_path, "audio/")
        relative_path = extractAfter(wav_path, "audio/");
    elseif contains(wav_path, "/audio/")
        relative_path = extractAfter(wav_path, "/audio/");
    else
        error('Unsupported audio path: %s', wav_path);
    end

    png_path = fullfile(spectrogram_folder, replace(relative_path, ".wav", ".png"));
end

function write_text_file(file_path, text_content)
    file_id = fopen(file_path, 'w');
    if file_id < 0
        error('Unable to write file: %s', file_path);
    end
    cleanup_obj = onCleanup(@() fclose(file_id));
    fwrite(file_id, text_content, 'char');
end
