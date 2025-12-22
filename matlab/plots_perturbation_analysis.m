clear all; clc; close all; 

set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',20)

% include yaml library to read the configuration file 
addpath("+yaml")

% custom colors 
nLines = 5;
blueShades = [linspace(0.5, 1, nLines)', linspace(0.4, 0.8, nLines)', linspace(1, 1, nLines)'];
blueShades(2, :) = blueShades(end-1, :);
blueShades(3:end, :) = 0.35*ones(3, 3);
%% LOAD DATA I

% semi-hardcoded directory names 
output_folder = "output/perturbation_test";

indx = 2;  % index for type_index  
type_indx = ["rt", "fc"];

test_folder = "perturb_b_"; 
output_file = "b_" + type_indx(indx);

test_folder = test_folder + type_indx(indx); 
loss_id = [1, 2, 3];
yml_file = "loss_profile_perturbation.yml";

% read config file
config = yaml.loadFile(fullfile("..", 'config', yml_file));
fs = 48000; 

% generate the steps dependingon the scale used 
if strcmp(config.param_config{indx, 1}.scale, 'linear')
    steps = linspace(config.param_config{indx, 1}.lower_bound,config.param_config{indx, 1}.upper_bound, config.param_config{indx, 1}.n_steps);
else
    steps = logspace(log10(config.param_config{indx, 1}.lower_bound), log10(config.param_config{indx, 1}.upper_bound), config.param_config{indx, 1}.n_steps);
    steps = steps ./ (2*pi) .* fs; 
end

% get the target parameter values in the correct display unit 
target = config.param_config{indx, 1}.target_value;
if indx == 2 
    target = target / 2 / pi * fs;
end
[~, target_indx] = min(abs(steps - target));  

%% LOAD DATA II

% load losses 
load(fullfile("..", output_folder, test_folder, "loss.mat"));
loss_name = string(loss_name);
loss_name_string = {'$\mathcal{L}_{\textrm{EDC,\,lin}}$', '$\mathcal{L}_{\textrm{EDC,\,log}}$', '$\mathcal{L}_{\textrm{MSS}}$'};

n_losses = length(loss_name);

losses = zeros(config.loss_config.n_runs, config.param_config{indx, 1}.n_steps, n_losses);

%% LOSS STANDARDIZATION 

for id = 1:length(loss_id)
    i = loss_id(id);
    temp = eval(loss_name(i));
    temp = reshape(temp, size(temp, 2), size(temp, 3), size(temp, 4));
    for j = 1:size(temp, 1)
        losses(j, :, i) = temp(j, :, :) - mean(temp(j, :, :), 'all');
        losses(j, :, i) = losses(j, :, i) ./ std(losses(j, :, i), 0, 'all');
    end
    temp = losses(:, :, i);
end

%% PLOT: MEAN LOSS

fig = figure('Name', 'loss_mean', 'Position', [60 10 1300 200*1.25]);
% Use tiled layout to minimize spacing between plots
if length(loss_id) > 3
    ncols = 5;
    nrows = floor(length(loss_id) / ncols);
    tl = tiledlayout(nrows, ncols, 'TileSpacing', 'tight', 'Padding', 'tight');
else
    tl = tiledlayout(1, length(loss_id), 'TileSpacing', 'compact', 'Padding', 'compact');
end

errors = []; 
for id = 1:length(loss_id)
    [~, min_idx] = min(losses(:, :, id), [], 2); 
    error = mean(abs(target - steps(min_idx))/target*100);
    errors = [errors, error];
    i = loss_id(id);

    % Create next plot
    nexttile(id);
    x = steps;
    
    % compute statistics 
    mu = squeeze(median(losses(:, :, i)));
    q_75 = squeeze(quantile(losses(:, :, i), 0.75, 1));
    q_25 = squeeze(quantile(losses(:, :, i), 0.25, 1));
    y1 = q_25;
    y2 = q_75;

    % plot 0.25-0.75 interval 
    fill([x, fliplr(x)], [y1, fliplr(y2)],  blueShades(1, :), 'EdgeColor', 'none', 'FaceAlpha', 0.3); % light blue 
    hold on

    % plot median 
    plot(steps, mu, 'Color', blueShades(1, :),'LineWidth', 1.5);

    view(0, 90)  % Azimuth = 0 degrees, Elevation = 90 degrees
    xlim([steps(1), steps(end)])
    grid on
    xline(target, ':', 'LineWidth', 1.5, 'Color', 'k')

    % labelling 
    title(loss_name_string{i})
    if strcmp(config.param_config{indx, 1}.scale, 'log')
        set(gca, 'XDir', 'normal','Xscale', 'log')
    end
    if indx == 1
        xlabel('$T_{60}$ (s)')
    else 
        set(gca, 'XTick', [1000 2000 4000 8000 16000]);
        set(gca, 'XTickLabel', {'1','2','4','8','16'}); % optional
        xlabel('$f_{\textrm{c}}$ (kHz)','Interpreter','latex')
    end
    if id == 1
        ylabel('Ampltidue')
    end
    
    % plot minimum
    min_val = squeeze(min(mu));
    [~, min_idx] = min(abs(squeeze(mu-min_val)));  
    hold on;
    plot(steps(min_idx), min_val, 'x', 'MarkerSize', 10, 'LineWidth', 2.5, 'Color', blueShades(1, :));
    hold off;
    ylim([-2.5, 4])
end

% Save the figure as EPS and FIG
set(fig, 'Color', 'w');
exportgraphics(fig, fullfile('eps', output_file + ".eps"), 'ContentType', 'vector');
% savefig(fig, fullfile("figs", output_file + ".fig"));


%% ACCURACY 
% note that this has not been used in the paper.

accuracy_ranges = [[75, 125]; [134, 184]];

accuracy = zeros(config.loss_config.n_runs, diff(accuracy_ranges(indx, 1:2))+2, n_losses);
for i_loss = 1:n_losses
    for i_run = 1:config.loss_config.n_runs
        accuracy(i_run, (accuracy_ranges(indx, 1):target_indx)-accuracy_ranges(indx, 1)+1, i_loss) = flip(diff(flip(losses(i_run, (accuracy_ranges(indx, 1)-1):target_indx, i_loss))));
        accuracy(i_run, (target_indx:accuracy_ranges(indx, 2))-accuracy_ranges(indx, 1)+2, i_loss) = diff(losses(i_run, target_indx:(accuracy_ranges(indx, 2)+1), i_loss));
    end
end

overall_accuracy = zeros(n_losses);
for i_loss = 1:n_losses
    overall_accuracy(i_loss) = mean(accuracy(:, :, i_loss )  > 0, "all");
end