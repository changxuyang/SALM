% Generating training dataset for SALM 1
clc; clear; close all;
rng(666);

%% Parameters
HR_SIZE         = [256, 256];    % high-resolution target size
LR_SIZE         = [86, 86];      % low-resolution input size
NUM_SAMPLES     = 500;           % number of samples
NUM_FRAMES      = 5;             % frames per sample 
NUM_SEGMENTS    = 5;             % segments per track (matches frames)
num_tracks      = randi([2, 15]);% track range per sample

% Save paths
h5Filename         = './SALM1_Dataset.H5';
Savepath_Input     = './Train/Input/';
Savepath_Structure = './Train/GT_Structure/';
Savepath_Velocity  = './Train/GT_Velocity/';
Savepath_Diection  = './Train/GT_Direction/';

% Ensure output folders exist
outDirs = {Savepath_Input, Savepath_Structure, Savepath_Velocity, Savepath_Diection};
for d = 1:numel(outDirs)
    if ~exist(outDirs{d}, 'dir'), mkdir(outDirs{d}); end
end
% Recreate H5 file if present
if exist(h5Filename, 'file'), delete(h5Filename); end

%% Pre-create HDF5 datasets
% /input: (W,H,C=1,Frames,Samples)
h5create(h5Filename, '/input',     [LR_SIZE(2), LR_SIZE(1), 1, NUM_FRAMES, NUM_SAMPLES], ...
    'Datatype', 'uint8', 'ChunkSize', [LR_SIZE(2), LR_SIZE(1), 1, NUM_FRAMES, 1], 'Deflate', 5);
% /structure, /velocity, /direction: (W,H,Samples)
h5create(h5Filename, '/structure', [HR_SIZE(2), HR_SIZE(1), NUM_SAMPLES], ...
    'Datatype', 'uint8', 'ChunkSize', [HR_SIZE(2), HR_SIZE(1), 1], 'Deflate', 5);
h5create(h5Filename, '/velocity',  [HR_SIZE(2), HR_SIZE(1), NUM_SAMPLES], ...
    'Datatype', 'uint8', 'ChunkSize', [HR_SIZE(2), HR_SIZE(1), 1], 'Deflate', 5);
h5create(h5Filename, '/direction', [HR_SIZE(2), HR_SIZE(1), NUM_SAMPLES], ...
    'Datatype', 'uint8', 'ChunkSize', [HR_SIZE(2), HR_SIZE(1), 1], 'Deflate', 5);

%% Sample generation
for imgIdx = 1:NUM_SAMPLES
    speedMapsAccum = cell(1, NUM_SEGMENTS); for s = 1:NUM_SEGMENTS, speedMapsAccum{s} = zeros(HR_SIZE); end
    direction_map  = zeros(HR_SIZE);
    inputFrames    = cell(1, NUM_FRAMES);   for f = 1:NUM_FRAMES,   inputFrames{f}   = zeros(HR_SIZE);   end
    wholeTrackMax  = zeros(HR_SIZE);
    GT             = zeros(HR_SIZE);

    % Tracks
    for tIdx = 1:num_tracks
        trackImg = zeros(HR_SIZE);
        
        % Track init
        sigma = 0.5 + rand() * 2.5;
        x = randi([25, 225]); y = randi([25, 225]); angle = 2 * pi * rand();

        % Power-law length
        xmin = 5; xmax = 200; alpha = 0.4;
        u = rand(1, 1);
        Ldraw = ((xmax^(1-alpha) - xmin^(1-alpha)) * u + xmin^(1-alpha)).^(1/(1-alpha));
        lengthh = floor(max(min(Ldraw, xmax), xmin));
        brightnessBaseline = 0.3 + 0.5*rand();
        points = []; brightness_values = [];

        % Random walk with boundary handling and angle jitter
        for j = 1:lengthh
            dx = round(cos(angle)); dy = round(sin(angle));
            if (dx == 0) && (dy == 0)
                direction = randi(4);
                switch direction
                    case 1, dx =  1; dy =  0;
                    case 2, dx = -1; dy =  0;
                    case 3, dx =  0; dy =  1;
                    case 4, dx =  0; dy = -1;
                end
            end
            if x + dx > HR_SIZE(1)
                ddrr = randi(2); dx = (-1)*(ddrr==1) + 0*(ddrr==2);
            end
            if x + dx == 0
                ddrr = randi(2); dx = ( 1)*(ddrr==1) + 0*(ddrr==2);
            end
            x = x + dx;

            if y + dy > HR_SIZE(1)
                ddrr = randi(2); dy = (-1)*(ddrr==1) + 0*(ddrr==2);
            end
            if y + dy == 0
                ddrr = randi(2); dy = ( 1)*(ddrr==1) + 0*(ddrr==2);
            end
            y = y + dy;

            angle = angle + (rand() * pi/6 - pi/12);
            angle = mod(angle, 2 * pi);

            brightness = normrnd(brightnessBaseline, brightnessBaseline/2, 1, 1);
            if brightness < 0, brightness = 0; end
            trackImg(x, y) = brightness;

            if isempty(points)
                points = [x, y]; brightness_values = [brightness];
            else
                if ~ismember([x, y], points, 'rows')
                    points = [points; x, y];
                    brightness_values = [brightness_values; brightness];
                end
            end
        end

        % Blur whole track and accumulate
        trackImgBlur  = imgaussfilt(trackImg, sigma);
        wholeTrackMax = max(wholeTrackMax, trackImgBlur);

        % Split the ordered track into 5 sequential groups
        num_points  = size(points, 1);
        avg_per_grp = num_points / NUM_SEGMENTS;
        min_per_grp = max(1, ceil(avg_per_grp * 0.2));
        max_per_grp = min(num_points, floor(avg_per_grp * 1.8));

        groups = cell(1, NUM_SEGMENTS); brightness_groups = cell(1, NUM_SEGMENTS);
        current_index = 1;
        for g = 1:NUM_SEGMENTS
            remaining_groups = NUM_SEGMENTS - g + 1;
            remaining_points = num_points - current_index + 1;
            if g == NUM_SEGMENTS
                groups{g}            = points(current_index:end, :);
                brightness_groups{g} = brightness_values(current_index:end, :);
                break;
            end
            max_points_current_group = min(max_per_grp, remaining_points - (remaining_groups - 1) * min_per_grp);
            num_points_current_group = randi([min_per_grp, max_points_current_group]);
            num_points_current_group = min(num_points_current_group, remaining_points);

            groups{g}            = points(current_index:current_index + num_points_current_group - 1, :);
            brightness_groups{g} = brightness_values(current_index:current_index + num_points_current_group - 1, :);
            current_index        = current_index + num_points_current_group;
        end

        % Build per-segment brightness maps and merge into frames
        segBrightImgs = cell(1, NUM_SEGMENTS);
        for s = 1:NUM_SEGMENTS, segBrightImgs{s} = zeros(HR_SIZE); end
        for s = 1:NUM_SEGMENTS
            pts = groups{s}; brs = brightness_groups{s};
            for k = 1:size(pts,1)
                segBrightImgs{s}(pts(k,1), pts(k,2)) = brs(k,1);
            end
            inputFrames{s} = max(inputFrames{s}, imgaussfilt(segBrightImgs{s}, sigma));
        end

        % Per-segment speed
        pixel_distanceWeight = 1.7; % distance per pixel weight
        for s = 1:NUM_SEGMENTS
            pts = groups{s};
            if size(pts,1) == 1
                total_distance = pixel_distanceWeight;
            else
                dxy = diff(pts, 1, 1);
                total_distance = sum(sqrt(sum(dxy.^2, 2))) * pixel_distanceWeight;
            end
            speed = total_distance;
            tmpSpeed = zeros(HR_SIZE);
            for l = 1:size(pts,1)
                tmpSpeed(pts(l,1), pts(l,2)) = speed;
                GT(pts(l,1), pts(l,2))       = 255;
            end
            speedMapsAccum{s} = max(speedMapsAccum{s}, tmpSpeed);
        end

        % Direction map
        for l = 1:size(points,1)-1
            dxx = points(l+1,1) - points(l,1);
            dyy = points(l+1,2) - points(l,2);
            direction = atan2(-dxx, dyy);
            if direction == 0
                direction = (dxx == 0 && dyy == 0) * 0 + (dxx ~= 0 || dyy ~= 0) * 10;
            end
            if direction < 0, direction = direction + 2*pi; end
            direction_map(points(l,1), points(l,2)) = direction;
        end
    end

    %% Downsample to LR and add Gaussian noise
    for s = 1:NUM_SEGMENTS
        inputFrames{s} = imresize(inputFrames{s}, LR_SIZE, 'nearest');
    end
    NoiseVar = 8e-3 * rand();
    for s = 1:NUM_SEGMENTS
        inputFrames{s} = uint8(255 * mat2gray(imnoise(mat2gray(inputFrames{s}), 'gaussian', 0, NoiseVar)));
    end

    % Save 5-page TIFF
    Savepath_TIFF = fullfile(Savepath_Input, sprintf('%05d', imgIdx) + ".tiff");
    imwrite(inputFrames{1}, Savepath_TIFF, 'WriteMode', 'overwrite');
    for s = 2:NUM_SEGMENTS
        imwrite(inputFrames{s}, Savepath_TIFF, 'WriteMode', 'append');
    end

    % Assemble /input tensor: (Frames,1,H,W) -> (W,H,1,Frames,1)
    inputSample = zeros(NUM_FRAMES, 1, LR_SIZE(1), LR_SIZE(2), 'uint8');
    for s = 1:NUM_SEGMENTS
        inputSample(s,1,:,:) = inputFrames{s};
    end

    speed_map_All = zeros(HR_SIZE);
    for s = 1:NUM_SEGMENTS
        speed_map_All = max(speed_map_All, speedMapsAccum{s});
    end
    imwrite(uint8(speed_map_All), fullfile(Savepath_Velocity, sprintf('%05d', imgIdx) + ".png"));

    % Save continuous direction map for reference (scaled to 0–255)
    imwrite(uint8(255*mat2gray(direction_map)), fullfile(Savepath_Diection, sprintf('%05d', imgIdx) + ".png"));

    % Discretize direction into {0,1,2,3,4}
    discrete_map = zeros(size(direction_map));
    for i = 1:size(direction_map,1)
        for j = 1:size(direction_map,2)
            ang = direction_map(i,j);
            if ang == 0
                discrete_map(i,j) = 0;
            else
                effective_ang = (ang == 10) * (2*pi) + (ang ~= 10) * ang;
                if effective_ang < 0, effective_ang = effective_ang + 2*pi; end
                if      effective_ang <=  pi/2,  discrete_map(i,j) = 1;
                elseif  effective_ang <=  pi,    discrete_map(i,j) = 2;
                elseif  effective_ang <= 3*pi/2, discrete_map(i,j) = 3;
                elseif  effective_ang <= 2*pi,   discrete_map(i,j) = 4;
                end
            end
        end
    end

    % Save structure GT
    GT_png = uint8(255 * mat2gray(GT));
    imwrite(GT_png, fullfile(Savepath_Structure, sprintf('%05d', imgIdx) + ".png"));

    % H5 layout
    inputSample   = permute(inputSample, [4, 3, 2, 1]); % (W,H,1,Frames)
    GT            = permute(GT,          [2, 1]);
    speed_map_All = permute(speed_map_All,[2, 1]);
    discrete_map  = permute(discrete_map, [2, 1]);

    inputSample   = reshape(inputSample,   [size(inputSample), 1]); % (W,H,1,Frames,1)
    GT            = uint8(reshape(GT,            [size(GT), 1]));
    speed_map_All = uint8(reshape(speed_map_All, [size(speed_map_All), 1]));
    discrete_map  = uint8(reshape(discrete_map,  [size(discrete_map),  1]));

    % Write H5
    h5write(h5Filename, '/input',     inputSample,   [1, 1, 1, 1, imgIdx], [LR_SIZE(2), LR_SIZE(1), 1, NUM_FRAMES, 1]);
    h5write(h5Filename, '/structure', GT,            [1, 1, imgIdx],       [HR_SIZE(2), HR_SIZE(1), 1]);
    h5write(h5Filename, '/velocity',  speed_map_All, [1, 1, imgIdx],       [HR_SIZE(2), HR_SIZE(1), 1]);
    h5write(h5Filename, '/direction', discrete_map,  [1, 1, imgIdx],       [HR_SIZE(2), HR_SIZE(1), 1]);

    if mod(imgIdx, 100) == 0
        fprintf('Generated %d samples\n', imgIdx);   
    end

    % Free large arrays
    clear direction_map discrete_map speed_map_All inputFrames speedMapsAccum segBrightImgs
end
