clear all; clc; close all; 

set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',20)

addpath("+yaml")
addpath("./Colormaps/")

% Define the two colors as RGB triplets (values between 0 and 1)
c2 = [254,97,0]./255;   % Color 1 (example)
c1 = [0.5 0.4 1];   % Color 2 (example)

nColors = 5; % Total number of colors in gradient

% Generate gradient
gradientColors = [linspace(c1(1), c2(1), nColors)', ...
                  linspace(c1(2), c2(2), nColors)', ...
                  linspace(c1(3), c2(3), nColors)'];
gradientColors(2, :) = gradientColors(end-1, :);
gradientColors(3:end, :) = 0.35*ones(3, 3);

%% LOAD DATA I

output_folder = "output/JAES-results";

test_folders = ["no_perturb_", "no_perturb_snr10_noiseinj_", "no_perturb_snr10_"]; % fig 8
% test_folders = ["no_perturb_noearly_", "no_perturb_noearly_snr10_noiseinj_", "no_perturb_noearly_snr10_"]; % fig 9
test_folders = ["no_perturb_mss_", "no_perturb_noisy_mss_"]; % fig 10

test_labels = ["$\mathcal{L}(h,\hat{h})$", "$\mathcal{L}(h + w_1,\hat{h} +w_2 )$", "$\mathcal{L}(h+w_1,\hat{h})$"]; 

indx = 1;
type_indx = ["rt", "fc"];
test_folders = test_folders + type_indx(indx); 
loss_id = [1, 2, 3];

% read config file
yml_file = "loss_profile_fig10_cpu.yml";
config = yaml.loadFile(fullfile("..", 'config', yml_file));
fs = 48000; 

% compute steps according to the required scale
if strcmp(config.param_config{indx, 1}.scale, 'linear')
    steps = linspace(config.param_config{indx, 1}.lower_bound,config.param_config{indx, 1}.upper_bound, config.param_config{indx, 1}.n_steps);
else
    steps = logspace(log10(config.param_config{indx, 1}.lower_bound), log10(config.param_config{indx, 1}.upper_bound), config.param_config{indx, 1}.n_steps);
    steps = steps ./ (2*pi) .* fs; 
end

% get target at the display unit 
target = config.param_config{indx, 1}.target_value;
if indx == 2 
    target = target / 2 / pi * fs;
end
%% LOAD DATA II

% plot 8 and 9 
loss_name_string = {'$\mathcal{L}_{\textrm{EDC,\,lin}}$', ...
'$\mathcal{L}_{\textrm{EDC,\,log}}$', ...
'$\mathcal{L}_{\textrm{MSS}}$'};

% plot 10 
loss_name_string = {'$\mathcal{L}_{\textrm{SC}}$', ...
 '$\mathcal{L}_{\textrm{SM}}$', ...
 '$\mathcal{L}_{\textrm{MSS}}$'};
load(fullfile("..", output_folder, test_folders(1), "loss.mat"));
loss_name_str = string(loss_name);
n_losses = length(loss_name);
losses = zeros(length(test_folders), config.loss_config.n_runs, config.param_config{indx, 1}.n_steps, n_losses);


for i_test = 1:length(test_folders)
    % load losses 
    load(fullfile("..", output_folder, test_folders(i_test), "loss.mat")); 
    for id = 1:length(loss_id)
        i = loss_id(id);
        temp = eval(loss_name_str(i)); % for some silly reason there's one more dim
        for j = 1:size(temp, 1)
            % Standardization - NOTE: no standardiation for plot 10 
            losses(i_test, j, :, i) = temp(j, :, :); % - mean(temp(j, :, :), 'all');
            losses(i_test, j, :, i) = losses(i_test, j, :, i); % ./ std(losses(i_test, j, :, i), 0, 'all');
        end
        temp = losses(i_test, :, :, i);
    end    
end

%% Firs channel loss LOSS
fig = figure('Name', 'loss_first_ch', 'Position', [60 10 1300 200*1.5]);
% Use tiled layout to minimize spacing between plots
tl = tiledlayout(1, length(loss_id), 'TileSpacing', 'compact', 'Padding', 'compact');

% create custom colors 
nLines = 5;
blueShades = [linspace(0.5, 1, nLines)', linspace(0.4, 0.8, nLines)', linspace(1, 1, nLines)'];
blueShades(2, :) = blueShades(end-1, :);
blueShades(3:end, :) = 0.35*ones(3, 3);
lineStyle = ["-", "-", "--", "-.", ":"];
lineWidth = [2.5, 2.5, 1, 1, 1];

hTests = gobjects(length(test_folders), 1);
axHandles = gobjects(length(loss_id), 1);
errors = [];
for i_test = length(test_folders):-1:1
    for id = 1:length(loss_id)
        i = loss_id(id);
        if i_test == length(test_folders)
            % Create the axes only once for each tile
            axHandles(id) = nexttile(id);
        end
        ax = axHandles(id);
        hold(ax,'on')
        h = plot(ax, steps, squeeze(losses(i_test, 1, :, i)), lineStyle(i_test), ...
                 'Color', gradientColors(i_test, :), 'LineWidth', lineWidth(i_test));
        [min_val, min_idx] = min(squeeze(losses(i_test, 1, :, i)));
        if i_test == 2
            error = abs(steps(min_idx) - target)/target * 100;
            errors = [errors, error];
        end
        plot(ax, steps(min_idx), min_val, 'x', 'MarkerSize', 10, 'LineWidth', lineWidth(i_test), 'Color', gradientColors(i_test, :))
        % store handle only when this is the first subplot (use id == 1)
        if id == 1
            hTests(i_test) = h;
            
        end
        if id == 1 && i_test == 1
            ylabel(ax, 'Amplitude')
        end
        view(0, 90)  % Azimuth = 0 degrees, Elevation = 90 degrees
        xlim([steps(1), steps(end)])
        grid on
        title(loss_name_string{i})
        if strcmp(config.param_config{indx, 1}.scale, 'log')
            set(ax, 'XDir', 'normal','Xscale', 'log')
        end
        if indx == 1
            xlabel('$T_{60}$ (s)')
        else 
            set(ax, 'XTick', [1000 2000 4000 8000 16000]);
            set(ax, 'XTickLabel', {'1','2','4','8','16'}); % optional
            xlabel('$f_{\textrm{c}}$ (kHz)','Interpreter','latex')
        end
        xline(target, ':', 'LineWidth', 1.5, 'Color', 'k')
        % ylim([-2.5, 4])

    end
end

%% 
errors
% Place a single legend above the tiled layout, outside the subplots
if exist('test_labels', 'var') && ~isempty(hTests)
    lg = legend(hTests, test_labels, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'FontSize', 18);
    lg.AutoUpdate = 'off';   % prevent legend from changing if you add more artists later
    % Attach the legend to the tiled layout as a tile at the top
    lg.Layout.Tile = 'north';
end