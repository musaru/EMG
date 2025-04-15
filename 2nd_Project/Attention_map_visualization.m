

% data_class_2 = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\CSV_file\class02_12ch.csv');
% data_class_30 = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\CSV_file\class30_12ch.csv');
% data_class_31 = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\CSV_file\class31_12ch.csv');


% % Separate time and channel data
% time = data_class_2(:, 1);              % First column is time
% channels = data_class_2(:, 2:end);      % Remaining 12 columns are EMG channels
% 
% figure;
% hold on;
% 
% % Define a custom set of 12 distinguishable colors
% customColors = [
%     1   0   0;      % red
%     0   1   0;      % green
%     0   0   1;      % blue
%     0   0   0;      % black
%     1   0.5 0;      % orange
%     0.5 0   0.5;    % purple
%     0   1   1;      % cyan
%      0.13 0.55 0.13;   % forest green 1   0   1;      % magenta
%     1.0 0.84 0;     % gold (replaces olive)
%     0.3 0.3 0.3;    % gray
%     0   0.5 0.5;    % teal
%     0.6 0.2 0.2     % brown
% ];
% % customColors = [
% %     1.00 0.00 0.00;   % red
% %     0.00 0.50 0.00;   % dark green
% %     0.00 0.00 0.55;   % dark blue
% %     0.00 0.00 0.00;   % black
% %     1.00 0.55 0.00;   % orange
% %     0.50 0.00 0.50;   % purple
% %     0.00 0.55 0.55;   % dark cyan
% %     0.55 0.00 0.55;   % dark magenta
% %     0.72 0.53 0.04;   % dark gold
% %     0.25 0.25 0.25;   % dark gray
% %     0.13 0.55 0.13;   % forest green
% %     0.40 0.26 0.13;   % dark brown
% % ];
% 
% 
% for i = 1:12
%     % Shadow effect: plot darker version behind the main line
%     %plot(time, channels(:,i), 'Color', customColors(i,:) * 0.5, 'LineWidth', 3); % shadow
%     % Main line
%     plot(time, channels(:,i), 'Color', customColors(i,:), 'LineWidth', 4, ...
%         'DisplayName', sprintf('Ch %d', i));
% end
% 
% hold off;
% xlabel('Time step');
% ylabel('Attention score');
% title('Channel Attention Weight');
% legend('Location', 'bestoutside');
% grid on;
% xlim([0 14500]);
% ylim([0.05 0.80]);




% === Read CSV files ===
%Time vs channel average drawing:
% data_class_2 = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\CSV_file\class02_12ch.csv');
% data_class_30 = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\CSV_file\class30_12ch.csv');
% data_class_31 = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\CSV_file\class31_12ch.csv');
% 
% % === Separate time and channel data ===
% time_2 = data_class_2(:, 1);
% channels_2 = data_class_2(:, 2:end);
% 
% time_30 = data_class_30(:, 1);
% channels_30 = data_class_30(:, 2:end);
% 
% time_31 = data_class_31(:, 1);
% channels_31 = data_class_31(:, 2:end);
% 
% % === Compute average across 12 channels ===
% channel_avg_2 = mean(channels_2, 2);
% channel_avg_30 = mean(channels_30, 2);
% channel_avg_31 = mean(channels_31, 2);
% 
% % === Plot the average signals ===
% thik=3
% figure;
% hold on;
% plot(time_2, channel_avg_2, 'r', 'LineWidth',thik, 'DisplayName', 'Class 02');    % Red
% plot(time_30, channel_avg_30, 'k', 'LineWidth', thik, 'DisplayName', 'Class 30');  % Black
% plot(time_31, channel_avg_31, 'b', 'LineWidth', thik, 'DisplayName', 'Class 31');  % Green
% hold off;
% 
% % === Add labels and styling ===
% xlabel('Time step');
% ylabel('Average Attention Score');
% title('Channel Average Comparison: Class 02, 30, and 31');
% legend('Location', 'best');
% grid on;
% xlim([0 14000]);
% %ylim([0.05 0.80]);



% intensity
% % === Read CSV files ===
% data_class_2  = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\CSV_file\class02_12ch.csv');
% data_class_30 = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\CSV_file\class30_12ch.csv');
% data_class_31 = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\CSV_file\class31_12ch.csv');
% 
% % === Separate time and channel data ===
% time_2  = data_class_2(:, 1);
% ch_2    = data_class_2(:, 2:end);
% 
% time_30 = data_class_30(:, 1);
% ch_30   = data_class_30(:, 2:end);
% 
% time_31 = data_class_31(:, 1);
% ch_31   = data_class_31(:, 2:end);
% 
% % channel_avg_2 = mean(channels_2, 2);
% % channel_avg_30 = mean(channels_30, 2);
% % channel_avg_31 = mean(channels_31, 2);
% 
% % === Plot Time vs Channel heatmaps for each class ===
% figure;
% 
% % Class 02
% subplot(1,3,1);
% imagesc(1:12, time_2, ch_2);
% colormap(jet);
% colorbar;
% title('Class 02');
% xlabel('Channel');
% ylabel('Time step');
% set(gca, 'YDir', 'normal');
% 
% % Class 30
% subplot(1,3,2);
% imagesc(1:12, time_30, ch_30);
% colormap(jet);
% colorbar;
% title('Class 30');
% xlabel('Channel');
% ylabel('Time step');
% set(gca, 'YDir', 'normal');
% 
% % Class 31
% subplot(1,3,3);
% imagesc(1:12, time_31, ch_31);
% colormap(jet);
% colorbar;
% title('Class 31');
% xlabel('Channel');
% ylabel('Time step');
% set(gca, 'YDir', 'normal');
% 
% sgtitle('Time vs Channel Intensity Map (12-channel EMG)');




%channel time wise average: 

data_class_2  = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\CSV_file\class02_12ch.csv');
data_class_30 = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\CSV_file\class30_12ch.csv');
data_class_31 = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\CSV_file\class31_12ch.csv');

% === Separate time and channel data ===
ch_2  = data_class_2(:, 2:end);   % 12 channels
ch_30 = data_class_30(:, 2:end);
ch_31 = data_class_31(:, 2:end);

% === Compute mean row for each class (mean across time) ===
mean_row_2  = mean(ch_2, 1);   % 1×12
mean_row_30 = mean(ch_30, 1);
mean_row_31 = mean(ch_31, 1);

% === Channel indices for x-axis ===
channels = 1:12;

% === Plot all mean rows ===
figure;
hold on;
plot(channels, mean_row_2,  '-o', 'LineWidth', 2, 'DisplayName', 'Class 02',  'Color', [1 0 0]);   % Red
plot(channels, mean_row_30, '-s', 'LineWidth', 2, 'DisplayName', 'Class 30',  'Color', [0 0 0]);   % Black
plot(channels, mean_row_31, '-^', 'LineWidth', 2, 'DisplayName', 'Class 31',  'Color', [0 0 1]);   % Blue
hold off;

% === Labels and formatting ===
xlabel('Channel Number');
ylabel('Mean Attention Score');
title('Mean Attention Score per Channel (Class 02, 30, 31)');
legend('Location', 'best');
grid on;
xlim([1 12]);
%ch_2    = data_class_2(:, 2:end);

% figure;
% hold on;
% for i = 1:12
%     plot(time, channels(:,i),'LineWidth', 4, 'DisplayName', sprintf('Channel %d', i));
% end
% hold off;
% xlabel('Time (s)');
% ylabel('Amplitude (µV)');
% title('Time vs EMG Amplitude (12 Channels)');
% legend show;
% grid on;
% 
% % Set axis limits
% xlim([0 14000]);
% ylim([0.05 0.80]);








files = {'class01_12ch.csv', 'class02_12ch.csv', 'class03_12ch.csv'};

for f = 1:length(files)
    data = readmatrix(files{f});
    time = data(:, 1);
    channels = data(:, 2:end);

    figure;
    hold on;
    for i = 1:12
        plot(time, channels(:, i), 'DisplayName', sprintf('Channel %d', i));
    end
    hold off;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(['File: ', files{f}]);
    legend show;
    grid on;
end





% % Read the CSV file
% % Load all three CSV files
% data_class_2  = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\Channel_average_files\class_02.csv');
% data_class_30 = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\Channel_average_files\class_30.csv');
% data_class_31 = readmatrix('G:\Professor_Work\EMG_Konnai\2nd_Paper\All Figure\Visualization\Attentin_matlab_figure\Channel_average_files\class_31.csv');
% 
% % Extract time and average values
% time_2  = data_class_2(:, 1);   avg_2  = data_class_2(:, 2);
% time_30 = data_class_30(:, 1);  avg_30 = data_class_30(:, 2);
% time_31 = data_class_31(:, 1);  avg_31 = data_class_31(:, 2);
% 
% % Plot
% figure;
% hold on;
% plot(time_2,  avg_2,  'r', 'LineWidth', 3, 'DisplayName', 'Class 02');
% plot(time_30, avg_30, 'k', 'LineWidth', 3, 'DisplayName', 'Class 30');
% plot(time_31, avg_31, 'g', 'LineWidth', 3, 'DisplayName', 'Class 31');
% hold off;
% 
% % Labels and formatting
% xlabel('Time (s)');
% ylabel('Average Amplitude (µV)');
% title('Time vs Average EMG Amplitude (3 Classes)');
% legend show;
% grid on;
% %xlim([0 14000]);
% %ylim([0.25 0.40]);

