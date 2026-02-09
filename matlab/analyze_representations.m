clear all; clc; close all; 

set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',26)

addpath("+yaml")
addpath("./Colormaps/")
mycolormap = customcolormap([0 .25 .5 .75 1], {'#9d0142','#f66e45','#ffffbb','#65c0ae','#5e4f9f'});

% colors 
customOrange = [254,97,0]./255;  
customPurple = [0.5 0.4 1];   
customGray = [0.15, 0.15, 0.15]; 
n_colors = 256;
rate = 0.125; % adjust this to control steepness (higher = steeper)
t = linspace(0, 1, n_colors);
exp_values = 4 * (exp(rate * t) - 1) / (exp(rate) - 1);

cm = [1 1 1; 0.9961 0.3804 0]; % [0 0 1; 1 1 1; 1 0 0]; 
mycolormapPosDiff = interp1([0; 4], cm, exp_values); 

rate = 0.25; % adjust this to control steepness (higher = steeper)
t = linspace(0, 1, n_colors);
exp_values = 14 * (exp(rate * t) - 1) / (exp(rate) - 1);
mycolormapPosDiff_II = interp1([0; 14], cm, exp_values); 

rate = 0.25; % adjust this to control steepness (higher = steeper)
t = linspace(0, 1, n_colors);
exp_values = 15 * (exp(rate * t) - 1) / (exp(rate) - 1);
mycolormapPosDiff_III = interp1([0; 15], cm, exp_values); 


t = linspace(0, 1, n_colors);
exp_values = 3 * (exp(rate * t) - 1) / (exp(rate) - 1);

cm = [1 1 1; 0.9961 0.3804 0]; % [0 0 1; 1 1 1; 1 0 0]; 
mycolormapPosDiff_IV = interp1([0; 3], cm, exp_values); 


cm = [1 1 1; 0.9961 0.3804 0; 0.5 0.4 1]; % [0 0 1; 1 1 1; 1 0 0]; 

%% LOAD DATA I

output_folder = "output/JAES-results";
test_folder = "target_validation";

load(fullfile("..", output_folder, test_folder, "fdn_edc.mat"))
load(fullfile("..", output_folder, test_folder, "target_edc.mat"))
load(fullfile("..", output_folder, test_folder, "fdn_stft.mat"))
load(fullfile("..", output_folder, test_folder, "target_stft.mat"))

% f_bands = [31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0];
% f_bands = [31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0];
f_bands = [31.5,    63. ,   125. ,   250. ,   500. ,  1000. ,  2000. ,4000. ,  8000. , 16000. ];
ir_len_edc = size(fdn_edc, 1);
ir_len = 48000*4;
fs = 48000; 


%% Plot the STFT linear
figure('Name', 'stft_linear', 'Position', [60 10 1300*0.9 700*0.9])

% plot spectrogram
nffts = [128, 256, 512, 1024, 2048, 4096]; 
keys = "nfft_" + string(nffts);
tiledlayout(2,3, 'TileSpacing','compact', 'Padding','compact')
for i_window = 1:length(nffts)

    cur_key = keys(i_window);
    target_stft_cur = target_stft.(cur_key);
    fdn_stft_cur = fdn_stft.(cur_key);

    time_axis = linspace(0, ir_len/fs, size(fdn_stft_cur, 2));
    freq_axis = linspace(0, fs/2, size(target_stft_cur, 1));

    ax = nexttile(i_window);
    pcol = pcolor(ax, time_axis, freq_axis,  abs(target_stft_cur - fdn_stft_cur));
    pcol.EdgeColor = 'none';
    set(ax, 'YDir', 'normal','Yscale', 'lin')
    % Set specific tick positions in Hz and label them in kHz
    yticks(ax, [1000, 5000, 10000, 15000, 20000]);
    yticklabels(ax, {'1', '5', '10', '15', '20'});
   
    if i_window > 3
        xlabel(ax, {'Time (s)'});
    end
    if i_window == 1 || i_window == 4
        ylabel(ax, "Frequency (kHz)")
    end
    ylim(ax, [31.5 20000])
    xlim(ax, [0, 1])
    title(nffts(i_window))
    colormap(mycolormapPosDiff)

    clb = colorbar;
    clb.TickLabelInterpreter = 'latex';

    set(ax, 'box', 'on', 'Visible', 'on')
    set(ax, 'layer','top')
    ax.FontSize = 26;                 % tick label font size
    ylabel(clb, 'Amplitude', 'Interpreter','latex');
    caxis(ax, [0, 4])
    % caxis([0, 8])
    if i_window ~= 3 && i_window ~= 6
        colorbar('off')
    end
end
exportgraphics(gcf, 'mss_abs_2.eps', 'ContentType', 'vector', 'Resolution', 300);
%% Plot the STFT logarithmic
figure('Name', 'stft_log', 'Position', [60 10 1300*0.9 700*0.9])


% plot spectrogram
nffts = [128, 512, 2048]; % [128, 256, 512, 1024, 2048, 4096]; 
keys = "nfft_" + string(nffts);
tiledlayout(2,3, 'TileSpacing','compact', 'Padding','compact')
 
for i_window = 1:length(nffts)
    cur_key = keys(i_window);
    target_stft_cur = target_stft.(cur_key);
    fdn_stft_cur = fdn_stft.(cur_key);

    time_axis = linspace(0, ir_len/fs, size(fdn_stft_cur, 2));
    freq_axis = linspace(0, fs/2, size(target_stft_cur, 1));

    ax = nexttile(i_window);
    pcol = pcolor(ax, time_axis, freq_axis,  abs(log(target_stft_cur) - log(fdn_stft_cur)));
    pcol.EdgeColor = 'none';
    set(ax, 'YDir', 'normal','Yscale', 'lin')
    yticks(ax, [1000, 5000, 10000, 15000, 20000]);
    yticklabels(ax, {'1', '5', '10', '15', '20'});
    if i_window > 3
        xlabel(ax, {'Time (s)'});
    end
    if i_window == 1 || i_window == 4
        ylabel(ax, "Frequency (kHz)")
    end
    ylim(ax, [31.5 20000])
    title(nffts(i_window))
    colormap(mycolormapPosDiff_II)
   
    clb = colorbar;
    clb.TickLabelInterpreter = 'latex';
    set(ax, 'box', 'on', 'Visible', 'on')
    set(ax, 'layer','top')
    ylabel(clb, 'Amplitude', 'Interpreter','latex');
    caxis(ax, [0, 14])
    % caxis([0, 8])
    if i_window ~= 3 && i_window ~= 6
        colorbar('off')
    end
end
for i_window = 1:length(nffts)
    cur_key = keys(i_window);
    target_stft_cur = target_stft.(cur_key);
    fdn_stft_cur = fdn_stft.(cur_key);

    time_axis = linspace(0, ir_len/fs, size(fdn_stft_cur, 2));
    freq_axis = linspace(0, fs/2, size(target_stft_cur, 1));

    ax = nexttile(i_window+3);
    pcol = pcolor(ax, time_axis, freq_axis,  abs(log(target_stft_cur) - log(fdn_stft_cur)));
    pcol.EdgeColor = 'none';
    set(ax, 'YDir', 'normal','Yscale', 'lin')
    xlabel(ax, {'Time (s)'});
    yticks(ax, [1000, 5000, 10000, 15000, 20000]);
    yticklabels(ax, {'1', '5', '10', '15', '20'});
    if i_window == 1 || i_window == 4
        ylabel(ax, "Frequency (kHz)")
    end
    ylim(ax, [31.5 20000])
    xlim(ax, [0 0.5])
    title(nffts(i_window))
    colormap(mycolormapPosDiff_II)
   
    clb = colorbar;
    clb.TickLabelInterpreter = 'latex';

    set(ax, 'box', 'on', 'Visible', 'on')
    set(ax, 'layer','top')
    ylabel(clb, 'Amplitude', 'Interpreter','latex');
    caxis(ax, [0, 14])
    % caxis([0, 8])
    if i_window ~= 3 
        colorbar('off')
    end
end

exportgraphics(gcf, 'mss_log_2.eps', 'ContentType', 'vector', 'Resolution', 100);
%% plot 2 
figure('Name', 'stft', 'Position', [60 10 1300*0.9 700*0.9])

tiledlayout(2,2, 'TileSpacing','compact', 'Padding','compact')
cur_key = keys(3);
target_stft_cur = target_stft.(cur_key);

fdn_stft_cur = fdn_stft.(cur_key);

time_axis = linspace(0, ir_len/fs, size(fdn_stft_cur, 2));
freq_axis = linspace(0, fs/2, size(target_stft_cur, 1));

% --- Tile 1 ---
ax1 = nexttile(1);
p1 = pcolor(ax1, time_axis, freq_axis, 20*log10(abs(target_stft_cur)));
p1.EdgeColor = 'none';
ax1.YDir = 'normal';
ax1.YScale = 'linear';           % or 'log' if you want log-scaled y-axis
caxis(ax1, [-100 10]);
title(ax1, '(a) $h(t) + w(t)$', 'Interpreter','latex')
yticks(ax1, [1000, 5000, 10000, 15000, 20000]);
yticklabels(ax1, {'1', '5', '10', '15', '20'});
ylabel(ax1, 'Frequency (kHz)')
cb1 = colorbar(ax1);             % per-tile colorbar (optional)
ylabel(cb1, 'Magnitude (dB)', 'Interpreter', 'Latex');
colormap(ax1, mycolormap)

% --- Tile 2 ---
ax2 = nexttile(2);
p2 = pcolor(ax2, time_axis, freq_axis, 20*log10(abs(fdn_stft_cur)));
p2.EdgeColor = 'none';
ax2.YDir = 'normal';
ax2.YScale = 'linear';   
caxis(ax2, [-100 10]);
yticks(ax2, [1000, 5000, 10000, 15000, 20000]);
yticklabels(ax2, {'1', '5', '10', '15', '20'});
title(ax2, '(b) $\hat{h}(t; \theta^*_{\Gamma})$', 'Interpreter','latex')
cb2 = colorbar(ax2); % right side of top row
ylabel(cb2, 'Magnitude (dB)', 'Interpreter', 'Latex');
colormap(ax2, mycolormap)

% --- Tile 3 ---
ax3 = nexttile(3);
p3 = pcolor(ax3, time_axis, freq_axis, abs(log(target_stft_cur) - log(fdn_stft_cur)));
p3.EdgeColor = 'none';
ax3.YDir = 'normal';
ax3.YScale = 'linear';
yticks(ax3, [1000, 5000, 10000, 15000, 20000]);
yticklabels(ax3, {'1', '5', '10', '15', '20'});
ylabel(ax3, 'Frequency (kHz)')
title(ax3, '(c) Logarithmic', 'Interpreter','latex')
xlabel(ax3, 'Time (s)')
cb3 = colorbar(ax3);
ylabel(cb3, '$\mathcal{J}_1$', 'Interpreter','latex');
colormap(ax3, mycolormapPosDiff_III)

% --- Tile 4 ---
ax4 = nexttile(4);
p4 = pcolor(ax4, time_axis, freq_axis, abs(target_stft_cur - fdn_stft_cur));
p4.EdgeColor = 'none';
ax4.YDir = 'normal';
ax4.YScale = 'linear';
yticks(ax4, [1000, 5000, 10000, 15000, 20000]);
yticklabels(ax4, {'1', '5', '10', '15', '20'});
title(ax4, '(d) Linear', 'Interpreter','latex')
xlabel(ax4, 'Time (s)')
cb4 = colorbar(ax4);
ylabel(cb4, '$\mathcal{J}_2$', 'Interpreter','latex');
colormap(ax4, mycolormapPosDiff_IV)

for ax = [ax1, ax2, ax3, ax4]
    ylim(ax, [31.5 20000])
    ax.FontSize = 30;
    set(ax, 'Box', 'on', 'Layer', 'top')
end

exportgraphics(gcf, 'spectrogram_analysis_2.eps', 'ContentType', 'vector', 'Resolution', 100);
%% Plot the EDC

target_edc_norm = target_edc - target_edc(1, :);
fdn_edc_norm = fdn_edc - fdn_edc(1, :);
time = [0:ir_len_edc-1]./fs;
figure('Name', 'edc', 'Position', [60 10 1500 700])
% Use a tight tiled layout instead of subplots
tl = tiledlayout(1, 4, 'TileSpacing', 'compact', 'Padding', 'tight');

lastAx = [];
legendHandles = gobjects(1, 2);
f_bands = [31.5,    63. ,   125. ,   250. ,   500. ,  1000. ,  2000. ,4000. ,  8000. , 16000. ];
indx = 3:2:10;
for i_indx = 1 : length(indx)
    i_band = indx(i_indx);
    ax = nexttile(i_indx); hold(ax, 'on'); grid(ax, 'minor'); 
    h1 = plot(ax, time, target_edc_norm(:, i_band), 'LineWidth', 3, 'Color', customGray); 
    h2 = plot(ax, time, fdn_edc_norm(:, i_band), 'LineWidth', 3, 'Color', customOrange);
    if i_indx == 1
        legendHandles = [h1, h2];
    end
    ylim(ax, [-100, 5])
    yline(ax, -60, '--', 'LineWidth', 1.5)
    xlim(ax, [0, 4])
    % Determine row and column position
    row = ceil(i_band / 5);
    col = mod(i_band - 1, 5) + 1;
    if i_indx == 1 
        ylabel(ax, "Energy (dB)")
    end
    if i_indx > 1
        yticklabels([]);
    end
    % Add band frequency as title
    title(ax, sprintf('%.1f Hz', f_bands(i_band)), 'Interpreter', 'latex');
    % Make tick labels smaller while keeping axis labels/titles readable
    ax.FontSize = 26;                 % tick label font size
    ax.XLabel.FontSize = 30;          % axis label font sizes
    ax.YLabel.FontSize = 30;
    ax.Title.FontSize = 30;           % slightly smaller title than labels
    set(ax, 'box', 'on', 'Visible', 'on')
    lastAx = ax;
end
xlabel(tl, 'Time (s)', 'Interpreter', 'latex', 'FontSize', 30);
% Single legend above all tiles using captured line handles
if all(isgraphics(legendHandles))
    lg = legend(legendHandles, {'$h(t) + w(t)$', '$\hat{h}(t; \theta^*_{\Gamma})$'}, 'Interpreter','latex', 'Orientation','horizontal', 'FontSize', 26);
    lg.Layout.Tile = 'north';
end



