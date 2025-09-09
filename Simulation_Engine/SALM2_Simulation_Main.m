% Generating training dataset for SALM 2
clc; clear; close all;
rng(666);

%% Parameters
HR_SIZE        = [256, 256];    % high-resolution target size
LR_SIZE        = [86, 86];      % low-resolution input size
NUM_SAMPLES    = 500;           % number of samples
NUM_FRAMES     = 5;             % frames per sample
NUM_SEGMENTS   = 5;             % segments per track (matches frames)
num_tracks     = randi([2, 15]);% track range per sample


% Save paths
h5Filename         = './SALM2_Dataset.H5';
Savepath_Input     = './Train/Input/';
Savepath_Structure = './Train/GT_Structure/';
Savepath_Velocity  = './Train/GT_Velocity/';
Savepath_Diection  = './Train/GT_Direction/';

% Ensure output folders exist
outDirs = {Savepath_Input, Savepath_Structure, Savepath_Velocity, Savepath_Diection};
for d = 1:numel(outDirs)
    if ~exist(outDirs{d}, 'dir'), mkdir(outDirs{d}); end
end

%% Pre-create HDF5 datasets
if exist(h5Filename, 'file')
    delete(h5Filename);
end

% /input stored as (W,H,C=1,Frames,Samples) so Python can read [N,5,1,H,W]
h5create(h5Filename, '/input', ...
    [LR_SIZE(2), LR_SIZE(1), 1, NUM_FRAMES, NUM_SAMPLES], ...
    'Datatype','uint8', 'ChunkSize',[LR_SIZE(2), LR_SIZE(1), 1, NUM_FRAMES, 1], 'Deflate',5);

% /structure, /velocity, /direction stored as (W,H,Samples)
h5create(h5Filename, '/structure', [HR_SIZE(2), HR_SIZE(1), NUM_SAMPLES], ...
    'Datatype','uint8', 'ChunkSize',[HR_SIZE(2), HR_SIZE(1), 1], 'Deflate',5);
h5create(h5Filename, '/velocity',  [HR_SIZE(2), HR_SIZE(1), NUM_SAMPLES], ...
    'Datatype','uint8', 'ChunkSize',[HR_SIZE(2), HR_SIZE(1), 1], 'Deflate',5);
h5create(h5Filename, '/direction', [HR_SIZE(2), HR_SIZE(1), NUM_SAMPLES], ...
    'Datatype','uint8', 'ChunkSize',[HR_SIZE(2), HR_SIZE(1), 1], 'Deflate',5);

%% Sample loop
for imgIdx = 1:NUM_SAMPLES
    speed_map       = zeros(HR_SIZE);
    speedMapsAccum  = cell(1, NUM_SEGMENTS);
    for s = 1:NUM_SEGMENTS, speedMapsAccum{s} = zeros(HR_SIZE); end
    direction_map = zeros(HR_SIZE);
    inputFrames = cell(1, NUM_FRAMES);      
    for f = 1:NUM_FRAMES, inputFrames{f} = zeros(HR_SIZE); end
    wholeTrackHR = zeros(HR_SIZE);         
    GT          = zeros(HR_SIZE);          

    % Store pulse-off positions
    pulsePos1 = cell(1, NUM_SEGMENTS);
    pulsePos2 = cell(1, NUM_SEGMENTS);



    %% Track loop
    for tIdx = 1:num_tracks
        trackImg = zeros(HR_SIZE);     
        sigma    = 0.5 + rand()*2.5;     % Gaussian blur sigma

        % Random start and angle
        x = randi([25, 225]);            
        y = randi([25, 225]);           
        angle = 2*pi*rand();

        % Power-law length
        xmin = 5; xmax = 200; alpha = 0.4;
        u = rand(1,1);
        Ldraw   = ((xmax^(1-alpha) - xmin^(1-alpha))*u + xmin^(1-alpha)).^(1/(1-alpha));
        lengthh = floor(max(min(Ldraw, xmax), xmin));

        brightnessBase = 0.3 + 0.5*rand();

        points = [];
        brightness_values = [];

        % Random walk with boundary handling and angle jitter
        for j = 1:lengthh
            dx = round(cos(angle));
            dy = round(sin(angle));

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

            if y + dy > HR_SIZE(2)
                ddrr = randi(2); dy = (-1)*(ddrr==1) + 0*(ddrr==2);
            end
            if y + dy == 0
                ddrr = randi(2); dy = ( 1)*(ddrr==1) + 0*(ddrr==2);
            end
            y = y + dy;

            % Angle jitter
            angle = angle + (rand()*pi/6 - pi/12);
            angle = mod(angle, 2*pi);

            % Brightness sample
            b1 = max(0, normrnd(brightnessBase, brightnessBase/2, 1, 1));
            trackImg(x, y) = b1;

            % Keep unique points in order
            if isempty(points)
                points = [x, y];
                brightness_values = b1;
            else
                if ~ismember([x, y], points, 'rows')
                    points            = [points; x, y];
                    brightness_values = [brightness_values; b1];
                end
            end
        end


        if ~isempty(points)
            GT(sub2ind(HR_SIZE, points(:,1), points(:,2))) = 255;
        end

        % Blur whole track
        trackImgBlur = imgaussfilt(trackImg, sigma);
        wholeTrackHR = max(wholeTrackHR, trackImgBlur);

        num_points  = size(points, 1);
        avg_per_grp = max(1, num_points / NUM_SEGMENTS);
        min_per_grp = max(1, ceil(avg_per_grp * 0.2));
        max_per_grp = min(num_points, floor(avg_per_grp * 1.8));

        groups = cell(1, NUM_SEGMENTS);
        brightness_groups = cell(1, NUM_SEGMENTS);
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

        % ---- Build Pulse-off positions ----
        for s = 1:NUM_SEGMENTS
            segBrightHR = zeros(HR_SIZE);
            pts = groups{s}; brs = brightness_groups{s};

            for k = 1:size(pts,1)
                segBrightHR(pts(k,1), pts(k,2)) = brs(k,1);
            end

            [pulsePos1{s}, pulsePos2{s}] = LightOFFpoint(pts);
            for m = 1:size(pulsePos1{s},1)
                segBrightHR(pts(pulsePos1{s}(m),1), pts(pulsePos1{s}(m),2)) = 0;
            end
            for m = 1:size(pulsePos2{s},1)
                segBrightHR(pts(pulsePos2{s}(m),1), pts(pulsePos2{s}(m),2)) = 0;
            end

            inputFrames{s} = max(inputFrames{s}, imgaussfilt(segBrightHR, sigma));
        end

        %% Speed maps
        PIXEL_DISTANCEWeight = 1.7;          % distance weight
        for s = 1:NUM_SEGMENTS
            pts = groups{s};
            P1  = pulsePos1{s};
            P2  = pulsePos2{s};

            if isempty(P1)
                if size(pts,1) == 1
                    total_distance = PIXEL_DISTANCEWeight;
                else
                    total_distance = sum(sqrt(sum(diff(pts).^2, 2))) * PIXEL_DISTANCEWeight;
                end

                for l = 1:size(pts,1)
                    speed_map(pts(l,1), pts(l,2)) = total_distance;
                end
            else

                sub1 = pts(1:P1(1), :);
                sub2 = pts(P1(end)+1:P2(1), :);
                sub3 = pts(P2(end)+1:end, :);

                if size(sub1,1) == 1
                    total_distance1 = PIXEL_DISTANCEWeight / (2.5/10);
                else
                    total_distance1 = sum(sqrt(sum(diff(sub1).^2, 2))) * PIXEL_DISTANCEWeight / (2.5/10);
                end
                if size(sub2,1) == 1
                    total_distance2 = PIXEL_DISTANCEWeight / (4.5/10);
                else
                    total_distance2 = sum(sqrt(sum(diff(sub2).^2, 2))) * PIXEL_DISTANCEWeight / (4.5/10);
                end
                if size(sub3,1) == 1
                    total_distance3 = PIXEL_DISTANCEWeight / (2.5/10);
                else
                    total_distance3 = sum(sqrt(sum(diff(sub3).^2, 2))) * PIXEL_DISTANCEWeight / (2.5/10);
                end



                for l = 1:size(sub1,1)
                    speed_map(sub1(l,1), sub1(l,2)) = total_distance1;
                end
                for l = 1:size(sub2,1)
                    speed_map(sub2(l,1), sub2(l,2)) = total_distance2;
                end
                for l = 1:size(sub3,1)
                    speed_map(sub3(l,1), sub3(l,2)) = total_distance3;
                end
            end

            speedMapsAccum{s} = max(speedMapsAccum{s}, speed_map);
            speed_map = zeros(HR_SIZE);
        end

        % Direction map
        for l = 1:size(points,1)-1
            dxx = points(l+1,1) - points(l,1);
            dyy = points(l+1,2) - points(l,2);

            ang = atan2(-dxx, dyy);
            if ang == 0
                if (dxx == 0) && (dyy == 0)
                    ang = 0;
                else
                    ang = 10;
                end
            end
            if ang < 0, ang = ang + 2*pi; end
            direction_map(points(l,1), points(l,2)) = ang;
        end
    end % track loop

    %% Downsample to LR and add Gaussian noise
    NoiseVar = 8e-3 * rand();
    for s = 1:NUM_SEGMENTS
        inputFrames{s} = imresize(inputFrames{s}, LR_SIZE, 'nearest');
        inputFrames{s} = uint8(255 * mat2gray(imnoise(mat2gray(inputFrames{s}), 'gaussian', 0, NoiseVar)));
    end

    % 5 frames: (Frames,1,H,W)
    inputSample = zeros(NUM_FRAMES, 1, LR_SIZE(1), LR_SIZE(2), 'uint8');
    for s = 1:NUM_SEGMENTS
        inputSample(s,1,:,:) = inputFrames{s};
    end

    % Merge speed maps over 5 frames
    speed_map_All = max(cat(3, speedMapsAccum{:}), [], 3);

    % Discretize direction into {0,1,2,3,4}
    discrete_map = zeros(size(direction_map));
    for i = 1:size(direction_map,1)
        for j = 1:size(direction_map,2)
            ang = direction_map(i,j);
            if ang == 0
                discrete_map(i,j) = 0;
            else
                eff_ang = (ang == 10) * (2*pi) + (ang ~= 10) * ang;
                if eff_ang < 0, eff_ang = eff_ang + 2*pi; end
                if      eff_ang <=  pi/2,  discrete_map(i,j) = 1;
                elseif  eff_ang <=  pi,    discrete_map(i,j) = 2;
                elseif  eff_ang <= 3*pi/2, discrete_map(i,j) = 3;
                elseif  eff_ang <= 2*pi,   discrete_map(i,j) = 4;
                end
            end
        end
    end

    %% Save images
    % 5-page TIFF for LR inputs
    tiffPath = fullfile(Savepath_Input, sprintf('%05d.tiff', imgIdx));
    imwrite(inputFrames{1}, tiffPath, 'WriteMode', 'overwrite');
    for s = 2:NUM_SEGMENTS
        imwrite(inputFrames{s}, tiffPath, 'WriteMode', 'append');
    end

    % Velocity PNG
    imwrite(uint8(speed_map_All), fullfile(Savepath_Velocity, sprintf('%05d.png', imgIdx)));

    % Continuous direction
    imwrite(uint8(255 * mat2gray(direction_map)), fullfile(Savepath_Diection, sprintf('%05d.png', imgIdx)));

    % Structure GT PNG
    GT_png = uint8(255 * mat2gray(GT));
    imwrite(GT_png, fullfile(Savepath_Structure, sprintf('%05d.png', imgIdx)));

    %% Layout and write to H5
    inputSample   = permute(inputSample, [4, 3, 2, 1]);   % -> (W,H,1,Frames)
    GT            = permute(GT,          [2, 1]);         % -> (W,H)
    speed_map_All = permute(speed_map_All,[2, 1]);        % -> (W,H)
    discrete_map  = permute(discrete_map, [2, 1]);        % -> (W,H)

    inputSample   = reshape(inputSample,   [size(inputSample), 1]); % (W,H,1,Frames,1)
    GT            = uint8(reshape(GT,            [size(GT), 1]));
    speed_map_All = uint8(reshape(speed_map_All, [size(speed_map_All), 1]));
    discrete_map  = uint8(reshape(discrete_map,  [size(discrete_map),  1]));

    h5write(h5Filename, '/input',     inputSample,   [1, 1, 1, 1, imgIdx], [LR_SIZE(2), LR_SIZE(1), 1, NUM_FRAMES, 1]);
    h5write(h5Filename, '/structure', GT,            [1, 1, imgIdx],       [HR_SIZE(2), HR_SIZE(1), 1]);
    h5write(h5Filename, '/velocity',  speed_map_All, [1, 1, imgIdx],       [HR_SIZE(2), HR_SIZE(1), 1]);
    h5write(h5Filename, '/direction', discrete_map,  [1, 1, imgIdx],       [HR_SIZE(2), HR_SIZE(1), 1]);

    if mod(imgIdx, 100) == 0
        fprintf('Generated %d samples\n', imgIdx);    
    end

    % Free large arrays
    clear speedMapsAccum direction_map discrete_map
end


function [indices_20_30, indices_60_80] = LightOFFpoint(points)
    N = size(points, 1); 
    if N <= 10
        indices_20_30 = [];
        indices_60_80 = [];
    else
        num_20_30 = max(1, round(N * 0.1)); 
        start_20_30 = round(0.2 * N + rand * 0.05 * N); 
        end_20_30 = start_20_30 + num_20_30 - 1; 
        indices_20_30 = (start_20_30:end_20_30)';

        num_60_80 = max(2, round(N * 0.2)); 
        start_60_80 = round(0.6 * N + rand * 0.1 * N); 
        end_60_80 = start_60_80 + num_60_80 - 1; 
        indices_60_80 = (start_60_80:end_60_80)';

        indices_20_30 = indices_20_30(indices_20_30 <= N);
        indices_60_80 = indices_60_80(indices_60_80 <= N);
    end
end
