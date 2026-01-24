%% Plot RIR waveform with onset marker for all analyzed files
matDir = fullfile(pwd, '');
wavDir = fullfile(pwd, '');

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
        x = mean(x, 2); % use mono
    end

    t = (0:numel(x)-1) ./ fs;

    figure('Name', baseName, 'NumberTitle', 'off');
    plot(t, x, 'b-');
    hold on;
    xline(onsetSample / fs, 'r--', 'LineWidth', 1.5);
    hold off;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('%s waveform with onset', baseName));
    legend({'Waveform', 'Onset'}, 'Location', 'best');
    grid on;
end
