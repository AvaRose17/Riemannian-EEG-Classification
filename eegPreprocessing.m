% Load the data

% this is the smaller dataset
% load('S1_run1.mat');

% full dataset
% Initialize an empty array or cell array depending on your data type
allData = [];

% Loop through each file
for i = 1:10
    % Construct the file name dynamically
    fileName = sprintf('S1_run%d.mat', i);
    
    % Check if file exists
    if isfile(fileName)
        % Load the file
        loadedData = load(fileName);
        
        % Check if the variable 'y' exists in the loaded file
        if isfield(loadedData, 'y')
            % Concatenate the data from each file into one larger array
            allData = [allData; loadedData.data]; % Adjust concatenation axis if needed
        else
            warning('Variable "y" not found in %s.', fileName);
        end
    else
        warning('%s does not exist.', fileName);
    end
end

% Display a small preview of the original data
%disp('Original data preview (first 10 timestamps and first 5 channels):');
%disp(y(1,1:10)); % Display the first 10 timestamps
%disp(y(2:6,1:10)); % Display the first 10 columns of the first 5 channels

% disp('last row of y (labels)');
% disp(y(258, :)); % display last row of y (should be the labels)

% PREPROCESSING 

disp("before downsampling");
disp(y(2:5, 1:10));

% Transpose the entire matrix to switch rows and columns, apply downsampling to rows (now time points), then transpose back
timestamps = y(1, :); % seperate timestamps
labels = y(258, :); % seperate event info
y = y(2:257, :);
y_transposed = y';  % Transpose the matrix to make time points as rows
y_downsampled_transposed = resample(y_transposed, 1, 3);  % Downsampling the transposed matrix
y_downsampled = y_downsampled_transposed';  % Transpose back to original orientation

disp("after downsampling");
disp(y_downsampled(2:5, 1:10));

% Separate the timestamps
% timestamps_downsampled = y_downsampled(1, :);  % This row now contains
% the downsampled timestamps % THIS IS AN ERROR I SHOULDN"T HAVE INCLUDED

% downsample the timestamps by keeping every third value
timestamps_downsampled = timestamps(1:3:end);

% downsampling the labels using majority rule
downsampling_factor = 3;
num_columns = size(y, 2);
downsampled_event_indices = ceil((1:num_columns) / downsampling_factor);
labels = accumarray(downsampled_event_indices.', labels', [], @mode).';


%disp("old data")
%disp(y(1:5, 1:10));
%disp("new downsampled data shoudln't include timesteps")
%disp(y_downsampled(1:5, 1:10));

% Separate the labels
% labels = y_downsampled(end, :);  % This row now contains the labels % ERROR I SHOULDN"T BE USING DOWNSAMPLED DATA

% ignore this for now
y_final = y_downsampled;  

% Display the size of the downsampled data without timestamps and labels
downsampled_size = size(y_final);
%fprintf('Downsampled size (without timestamps and labels): %d rows, %d columns\n', downsampled_size(1), downsampled_size(2));

% Display the size of the downsampled timestamps
fprintf('Length of downsampled timestamps: %d\n', length(timestamps_downsampled));
%disp(timestamps_downsampled(1:5));

% Display the size of the labels
%fprintf('Length of labels: %d\n', length(labels));

% Sampling frequency
Fs = 200;

% Fundamental frequency of powerline noise
f0 = 60;  % Fundamental frequency (60 Hz)

% Design and apply notch filters iteratively for 60 Hz and its harmonics
for harmonic = 1:floor(Fs/(2*f0))
    fn = f0 * harmonic;  % Harmonic frequency
    if fn >= Fs/2
        break;
    end

    % Design a 4th-order Butterworth notch filter
    % Notch range: fn +/- 1 Hz
    [b, a] = butter(4, [(fn - 1)/(Fs/2), (fn + 1)/(Fs/2)], 'stop');

    % Apply the filter using filtfilt for zero-phase filtering
    y_downsampled = filtfilt(b, a, y_downsampled);
end

% Apply a 4th-order Butterworth band-pass filter from 8 to 25 Hz
[b, a] = butter(4, [8 25]/(Fs/2), 'bandpass');
y_filtered = filtfilt(b, a, y_downsampled);


% Common average reference the filtered data
y_car = y_filtered - mean(y_filtered, 1);

% Identify bad channels based on the power criteria
power_threshold = 6;  % Z-score threshold for bad channels
channel_powers = mean(y_car.^2, 2);  % Mean power of each channel
log_channel_powers = log(channel_powers);  % Log transform for normality
z_scores = zscore(log_channel_powers);  % Z-score transformation
bad_channels = find(z_scores > power_threshold);

% Identify runs with more than 10% bad channels as bad runs
num_channels = size(y_car, 1);
bad_runs = sum(z_scores > power_threshold) > 0.1 * num_channels;

% Apply a disjunction to bad channels over the remaining runs
% If a channel is considered bad in one run, it is considered bad in all runs
for r = find(bad_runs)
    y_car(bad_channels, r) = NaN;  % Remove bad channels data
end

% Final common average referencing of the cleaned data
y_final = y_car - mean(y_car, 1, 'omitnan');

% Print summary information about the processing
disp('Processing complete. Bad channels and runs identified and managed.');

disp(y_final(1:5,1:10)); % Display the first 10 columns of the first 5 channels


% FEATURE EXTRACTION AND EPOCHING

% Set the sampling frequency
Fs = 200;

% Define the frequency bands for mu and beta
mu_band = [8 13];
beta_band = [13 30];

% Design band-pass filters for mu and beta bands using the butter function
[b_mu, a_mu] = butter(4, mu_band/(Fs/2), 'bandpass');
[b_beta, a_beta] = butter(4, beta_band/(Fs/2), 'bandpass');

% Apply the band-pass filters
y_mu = filtfilt(b_mu, a_mu, y_final);
% disp("y_mu size");
% disp(size(y_mu)) % same size as y_final but non-mu amplitudes are reduced
y_beta = filtfilt(b_beta, a_beta, y_final);

% Calculate the power of the filtered signals
power_mu = y_mu.^2;
power_beta = y_beta.^2;

% Segment and average the power in non-overlapping 0.25 s segments
segment_length = Fs * 0.25;  % 0.25 seconds * sampling frequency

% Number of complete segments that can fit in the signal
num_segments_mu = floor(size(power_mu, 2) / segment_length);
num_segments_beta = floor(size(power_beta, 2) / segment_length);

% Calculate the labels to correspond to newly segmented data
% Example: Assigning labels based on majority voting (Experiment with other
% options!)
segment_labels = zeros(1, num_segments_mu);  % Assuming num_segments is defined
for i = 1:num_segments_mu
    segment_start = (i - 1) * segment_length + 1;
    segment_end = segment_start + segment_length - 1;
    segment_labels(i) = mode(labels(segment_start:segment_end));
end


% Initialize matrices to store average power per segment
mu_power_segments = zeros(size(y_mu, 1), num_segments_mu);
beta_power_segments = zeros(size(y_beta, 1), num_segments_beta);

% Calculate average power for each segment
for i = 1:num_segments_mu
    start_idx = (i - 1) * segment_length + 1;
    end_idx = start_idx + segment_length - 1;
    mu_power_segments(:, i) = mean(power_mu(:, start_idx:end_idx), 2);
end

for i = 1:num_segments_beta
    start_idx = (i - 1) * segment_length + 1;
    end_idx = start_idx + segment_length - 1;
    beta_power_segments(:, i) = mean(power_beta(:, start_idx:end_idx), 2);
end

% Optionally, later apply a moving average to smooth the segment power values
% I DELETED MOVING AVERAGE CALCULATION. It was creating too many errors...

% CLASSIFICATION

% Combine features into a single matrix (mu and beta power segments)
features = [mu_power_segments', beta_power_segments']; % Horizontal concatenation

disp("size mu power")
disp(size(mu_power_segments))
disp("size beta power")
disp(size(beta_power_segments))
disp("size features")
disp(size(features));
disp("segmented labels")
disp(size(segment_labels))

% Train and cross-validate the SVM Classifier for multi-class classification
SVMModel = fitcecoc(features, segment_labels, 'Learners', 'linear', 'Coding', 'onevsall', ...
    'CrossVal', 'on', 'KFold', 10);

% Calculate the cross-validated classification accuracy
classificationAccuracy = 1 - kfoldLoss(SVMModel, 'LossFun', 'ClassifError');
fprintf('Cross-validated classification accuracy: %.2f%%\n', classificationAccuracy * 100);

% Predict labels for the cross-validated models
predictedLabels = kfoldPredict(SVMModel);

% Generate a confusion matrix
confMat = confusionmat(segment_labels, predictedLabels);
figure;
confusionchart(confMat);
title('Confusion Matrix for SVM Classification');
