clear all; clc; close all; 

set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',20)

addpath("+yaml")
addpath("./Colormaps/")

% Option to use actual minima from loss surface instead of saved minima
use_actual_minima = true;  % Set to true to compute minima from curr_loss

mycolormap = customcolormap([0 .25 .5 .75 1], {'#9d0142','#f66e45','#ffffbb','#65c0ae','#5e4f9f'});

cm = [0.5 0.4 1; 1 1 1; 0.9961 0.3804 0 ]; % [0 0 1; 1 1 1; 1 0 0]; 
cmi = interp1([-5; 0; 5], cm, (-5:0.1:5)); 

%% LOAD DATA I

output_folder = "output/";
test_folders = ["no_perturb_no_noise","no_perturb_2d_aware", "no_perturb_2d_agnostic"];
loss_id = [1, 2, 3];
loss_name_string = {'$\mathcal{L}_{\textrm{EDC,\,lin}}$', ...
'$\mathcal{L}_{\textrm{EDC,\,log}}$', ...
'$\mathcal{L}_{\textrm{MSS}}$'};

% read config file
yml_file = "loss_surface_gpu.yml";
config = yaml.loadFile(fullfile("..", 'config', yml_file));
fs = 48000; 

% compute steps according to the required scale
steps_1 = linspace(config.param_config{1, 1}.lower_bound,config.param_config{1, 1}.upper_bound, config.param_config{1, 1}.n_steps);
steps_2 = logspace(log10(config.param_config{2, 1}.lower_bound), log10(config.param_config{2, 1}.upper_bound), config.param_config{2, 1}.n_steps);
steps_2 = steps_2 ./ (2*pi) .* fs; 

% get target at the display unit 
targets = [config.param_config{1, 1}.target_value, config.param_config{2, 1}.target_value / 2 / pi * fs ];

saved_minima_first = load("min_values_first_ch.mat");
saved_minima_second = load("min_values_second_ch.mat");

%% LOAD DATA I

for i_test = 1:length(test_folders)
    load(fullfile("..", output_folder, test_folders(i_test), "loss.mat"));
    loss = squeeze(loss);
    figure('Position', [60 10 1300 200*1.25]);
    tl = tiledlayout(1, 3, 'TileSpacing', 'tight', 'Padding', 'tight');
    for i_loss = 1:length(loss_id)
        nexttile(i_loss)
        curr_loss = loss(:, :, i_loss);
        curr_loss = (curr_loss - mean(curr_loss,"all")) ./ std(curr_loss, 0, 'all');
        curr_loss = curr_loss.';
        surf(steps_1, steps_2, curr_loss, 'EdgeColor', 'none', 'FaceAlpha',1);
        
        % Get minima either from actual loss surface or from saved values
        if use_actual_minima
            % Find minimum in the current loss surface
            [~, min_idx] = min(curr_loss(:));
            [min_step_2_idx, min_step_1_idx] = ind2sub(size(curr_loss), min_idx);
            min_step_1 = steps_1(min_step_1_idx);
            min_step_2 = steps_2(min_step_2_idx);
        else
            % Use saved minima from previous analysis
            min_step_1 = saved_minima_first.min_steps(i_test, i_loss);
            min_step_2 = saved_minima_second.min_steps(i_test, i_loss);
        end
        
        hold on
        plot3(min_step_1, min_step_2, max(curr_loss, [], 'all') + 1, 'kx', 'MarkerSize', 12, 'LineWidth', 1.5)
        view(0, 90)  % Azimuth = 0 degrees, Elevation = 90 degrees
        if i_loss == length(loss_id)
            colorbar
        end
        clim([-2 4])
        xlim([steps_1(1, 1), steps_1(end, end)])
        ylim([steps_2(1, 1), steps_2(end, end)])
        set(gca,'yscale','log')
        set(gca, 'YTick', [1000 2000 4000 8000 16000]);
        set(gca, 'YTickLabel', {'1','2','4','8','16'}); % optional
        grid on
        colormap(mycolormap); 
        title(loss_name_string(i_loss));
        xline(targets(1), ':', 'LineWidth', 1.5, 'Color', 'k')
        yline(targets(2), ':', 'LineWidth', 1.5, 'Color', 'k')
        xlabel('$T_{\textrm{60}}$ (s)','Interpreter','latex')
        if i_loss == 1
            ylabel('$f_{\textrm{c}}$ (kHz)','Interpreter','latex')
        else
            ylabel('')
        end
    end
    exportgraphics(gcf, test_folders(i_test) + '.eps', 'ContentType', 'vector', 'Resolution', 100);
end