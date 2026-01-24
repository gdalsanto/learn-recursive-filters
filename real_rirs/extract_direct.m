clear all; close all; clc

fs = 48000; % desired sampling frequency
[data, filenames] = load_wav_files();

rir_struct = struct();


% check that the audio files have the correct sampling frequency, and resample if necessary
for i = 1:size(data, 1)
    audio = data{i, 1};
    audio_fs = data{i, 2};
    if audio_fs ~= fs
        fprintf('Resampling audio file %d from %d Hz to %d Hz\n', i, audio_fs, fs);
        audio = resample(audio, fs, audio_fs);
        data{i, 1} = audio;
        data{i, 2} = fs;
    end
    rir_struct.(filenames(i)).onset_time = 0;
    rir_struct.(filenames(i)).noise = 0;
    rir_struct.(filenames(i)).rt = 0;
end
%% 
% plot the data 
figure;
for i = 1:size(data, 1)
    subplot(size(data, 1), 1, i);
    plot((data{i, 1}));
    title(sprintf('Audio File %d', i));
end

% hardcoded onset time 
onset_times = []; 

% hardcoded background noise 




%% 
% Store in structure



figure;x
for i = 1:size(data, 1)
    subplot(size(data, 1), 1, i);
    [edc, time_ms] = schroeder_backward_integration(data{i, 1}, fs);
    plot(time_ms, edc);
    title(sprintf('Audio File %d', i));
end


function [audio_data, filenames ]= load_wav_files(folder_path)
    % Load all WAV files from a specified folder
    % Input: folder_path - path to folder containing WAV files
    % Output: audio_data - cell array containing audio data and sample rates
    
    if nargin < 1
        folder_path = pwd; % Use current folder if not specified
    end
    
    % Get list of all WAV files
    wav_files = dir(fullfile(folder_path, '*.wav'));
    
    if isempty(wav_files)
        warning('No WAV files found in folder: %s', folder_path);
        audio_data = {};
        return;
    end
    
    % Initialize cell array
    audio_data = cell(length(wav_files), 2);
    filenames = [];
    % Load each WAV file
    for i = 1:length(wav_files)
        file_path = fullfile(wav_files(i).folder, wav_files(i).name);
        filenames = [filenames; wav_files(i).name];
        [audio, fs] = audioread(file_path);
        audio_data{i, 1} = audio;
        audio_data{i, 2} = fs;
        fprintf('Loaded: %s (Fs = %d Hz)\n', wav_files(i).name, fs);
    end
    
    fprintf('Total files loaded: %d\n', length(wav_files));
end

function [edc, time_ms] = schroeder_backward_integration(audio, fs)
    % Compute broadband Schroeder backward integration (Energy Decay Curve)
    % Input: audio - audio signal (column vector)
    %        fs - sampling frequency (Hz)
    % Output: edc - energy decay curve (normalized, in dB)
    %         time_ms - time vector (milliseconds)
    
    % Ensure audio is a column vector
    if isrow(audio)
        audio = audio';
    end
    
    % Compute squared signal (energy)
    energy = audio.^2;
    
    % Backward integration
    edc = flipud(cumsum(flipud(energy)));
    
    % Normalize to start at 0 dB
    edc = edc / max(edc);
    edc = db(edc);
    
    % Create time vector (milliseconds)
    time_samples = length(edc);
    time_ms = (0:time_samples-1)' / fs * 1000;
end