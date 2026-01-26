close all; clear all; clc 
%% Plot RIR waveform with onset marker for all analyzed files
matDir = fullfile(pwd, 'sdm');
wavDir = fullfile(pwd, 'sdm');

matFiles = dir(fullfile(matDir, '*_analysis.mat'));
if isempty(matFiles)
    error('No analysis files found in %s', matDir);
end

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

    subplot(2, 1, 1);
    plot(t, x, 'b-');
    hold on;
    xline(onsetSample / fs, 'r--', 'LineWidth', 1.5);
    hold off;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('%s waveform with onset', strrep(baseName, '_', ' ')));
    legend({'Waveform', 'Onset'}, 'Location', 'best');
    grid on;

    subplot(2, 1, 2);
    plot(double(data.bands), double(data.rt60), 'b-');
    xlabel('Frequency (Hz)');
    ylabel('RT60 (s)');
    title(sprintf('%s rt60', strrep(baseName, '_', ' ')));
    grid on;
    close all


    % extract noise floor 
    start_noise = int64(data.noise_floor_samps) + data.onset;
    noise_floor = x(start_noise:end);
    snr = mean(abs(x).^2) / mean(abs(noise_floor).^2);
    disp(baseName)
    disp("snr: " + num2str(snr));
end
