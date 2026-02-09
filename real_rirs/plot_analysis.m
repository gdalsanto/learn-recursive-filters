close all; clear all; clc 
%% Plot RIR waveform with onset marker for all analyzed files
matDir = fullfile(pwd, 'sdm_selected');
wavDir = fullfile(pwd, 'sdm_selected');

matFiles = dir(fullfile(matDir, '*_analysis.mat'));
if isempty(matFiles)
    error('No analysis files found in %s', matDir);
end

freq_bands = [125, 250, 500, 1000, 2000, 4000]; 


for k = 1:numel(matFiles)
    matPath = fullfile(matDir, matFiles(k).name);
    data = load(matPath);

    onsetSample = double(data.onset);
    onsetSample = onsetSample(:);
    onsetSample = onsetSample(1); % use the first value if array-like

    baseName = erase(matFiles(k).name, '_analysis.mat');
    wavPath = fullfile(wavDir, baseName + ".wav");

    if ~isfile(wavPath)
        warning('Missing wav file for %s (expected %s)', matFiles(k).name, wavPath);
        continue
    end

    [x, fs] = audioread(wavPath);
    if size(x, 2) > 1
        x = x(:, 1); % use mono
    end

    t = (0:numel(x)-1) ./ fs;

    figure('Name', baseName, 'NumberTitle', 'off');

    subplot(3, 1, 1);
    plot(t, x, 'b-');
    hold on;
    xline(onsetSample / fs, 'r--', 'LineWidth', 1.5);
    hold off;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('%s waveform with onset', strrep(baseName, '_', ' ')));
    legend({'Waveform', 'Onset'}, 'Location', 'best');
    grid on;

    subplot(3, 1, 2);
    plot(double(data.bands), double(data.rt60), 'b-');
    hold on 
    plot(double(data.bands), data.dfn_rt60, 'r-' )
    xlabel('Frequency (Hz)');
    ylabel('RT60 (s)');
    title(sprintf('%s rt60', strrep(baseName, '_', ' ')));
    legend('reg', 'dfn')
    grid on;

    subplot(3, 1, 3);
    plot(double(data.bands), data.dfn_amp, 'r-' )
    xlabel('Frequency (Hz)');
    ylabel('RT60 (s)');
    title(sprintf('%s rt60', strrep(baseName, '_', ' ')));
    grid on;

    % extract noise floor 
    start_noise = int64(data.noise_floor_samps) + data.onset;
    noise_floor = x(start_noise:end);
    snr = mean(abs(x).^2) / mean(abs(noise_floor).^2);
    disp(baseName)
    disp("snr: " + num2str(snr));


end
