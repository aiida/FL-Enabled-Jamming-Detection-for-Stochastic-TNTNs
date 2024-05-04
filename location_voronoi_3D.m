% Function to initial locations of UEs and UAV CHs and Jammers
% Input:
%   noUsers   == double == number of active UEs
%   noSBS     == double == number of SBSs
%   flag_plot == bollean, = 1 to plot to test, = 0 to skip
function [UE_BS, JAM_BS, UEs, JAMs, BS] = location_voronoi_3D(num_cells, flag_plot)

% Initialization

%% Simulation window parameters
xMin=0;
xMax=55;
yMin=0;
yMax=55;
zMin = 0; 
zMax = 10; % Adding z-dimension for UAV CHs

xDelta=xMax-xMin; %width 
yDelta=yMax-yMin; %height

% Define network size for a 3D space
network_volume = xDelta * yDelta * (zMax - zMin); % Volume of simulation space


%% Active Users

lambda_U = 0.0015;
num_users=poissrnd(network_volume*lambda_U);%Poisson number of points
N_active_ue = num_users;
pos_users = [xDelta * rand(num_users, 1), yDelta * rand(num_users, 1), zeros(num_users, 1)]; % Users are at ground level (z=0);

UEs.active = pos_users; % Active users

%% Inactive users

N_inactive_ue = 2;
UEs.inactive = [xDelta * rand(N_inactive_ue, 1), yDelta * rand(N_inactive_ue, 1), zeros(N_inactive_ue, 1)]; % Inactive users also at ground level

%% Active jammers

lambda_J = 0.00037;
num_jammers=poissrnd(network_volume*lambda_J);%Poisson number of points
N_active_jam = num_jammers;
pos_jammers = [xDelta * rand(num_jammers, 1), yDelta * rand(num_jammers, 1), zeros(num_jammers, 1)]; % Jammers at ground level

JAMs.active = pos_jammers;  % [m] x-coordinate of points

%% Inactive jammers

N_inactive_jam = 1;
JAMs.inactive = [xDelta * rand(N_inactive_jam, 1), yDelta * rand(N_inactive_jam, 1), zeros(N_inactive_jam, 1)]; % Inactive jammers


%% UAV CHs

pos_CHs = [xDelta * rand(num_cells, 1), yDelta * rand(num_cells, 1), zMin + (zMax - zMin) * rand(num_cells, 1)]; % UAVs are at various heights above ground
BS.positions = pos_CHs;


%% Plotting Location of Points Inside 2D Circular Region
if flag_plot
    figure;
    hold on;
    grid on;
    xlabel('x [m]', 'Interpreter', 'latex');
    ylabel('y [m]', 'Interpreter', 'latex');
    zlabel('z [m]', 'Interpreter', 'latex');

%     plot(mbs_center(1), mbs_center(2),'md','MarkerFaceColor','m', 'HandleVisibility','off'); hold on;                % Location of points
%     text(mbs_center(:,1)+13, mbs_center(:,2), 'PS', ...
%        'HorizontalAlignment','left')

    % Plot active users
    scatter3(UEs.active(:,1), UEs.active(:,2), UEs.active(:,3), 'bo', 'Filled', 'DisplayName', 'Active Users');

    % Plot inactive users
    scatter3(UEs.inactive(:,1), UEs.inactive(:,2), UEs.inactive(:,3), 'mo', 'Filled', 'DisplayName', 'Inactive Users');

    % Plot active jammers
    scatter3(JAMs.active(:,1), JAMs.active(:,2), JAMs.active(:,3), 'ro', 'Filled', 'DisplayName', 'Active Jammers');

    % Plot inactive jammers
    scatter3(JAMs.inactive(:,1), JAMs.inactive(:,2), JAMs.inactive(:,3), 'co', 'Filled', 'DisplayName', 'Inactive Jammers');

    % Plot UAV CHs
    scatter3(BS.positions(:,1), BS.positions(:,2), BS.positions(:,3), 'gs', 'Filled', 'DisplayName', 'UAV CHs');

    for i = 1:num_cells
        % Generate label for each CH
        label = sprintf('CH_%d', i);
        
        % Get the position for placing the label. Slightly offset the label in z-direction for visibility
        x = BS.positions(i,1);
        y = BS.positions(i,2);
        z = BS.positions(i,3) + 1; % Adjust the 1 if necessary for better visibility
        
        % Place the label
        text(x, y, z, label, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'Interpreter', 'latex');
    end
    
    % Draw lines from users to their nearest UAV CH
    for i = 1:size(UEs.active, 1)
        [~, nearestCH] = min(sum((UEs.active(i,1:3) - BS.positions).^2, 2));
        line([UEs.active(i,1), BS.positions(nearestCH,1)], ...
             [UEs.active(i,2), BS.positions(nearestCH,2)], ...
             [UEs.active(i,3), BS.positions(nearestCH,3)], 'Color', 'b', 'LineStyle', '--');
    end

    % Draw lines from jammers to their nearest UAV CH
    for j = 1:size(JAMs.active, 1)
        [~, nearestCH] = min(sum((JAMs.active(j,1:3) - BS.positions).^2, 2));
        line([JAMs.active(j,1), BS.positions(nearestCH,1)], ...
             [JAMs.active(j,2), BS.positions(nearestCH,2)], ...
             [JAMs.active(j,3), BS.positions(nearestCH,3)], 'Color', 'r', 'LineStyle', '--');
    end
    
    legend;
    view(3); % Set the view to 3D
end

%% 
% Calculate nearest UAV CH for each User and Jammer
UE_BS = zeros(num_users, num_cells);
JAM_BS = zeros(num_jammers, num_cells);

for i = 1:num_users
    [~, nearestCH] = min(sum((UEs.active(i,1:3) - BS.positions).^2, 2));
    UE_BS(i, nearestCH) = 1;
end

for j = 1:num_jammers
    [~, nearestCH] = min(sum((JAMs.active(j,1:3) - BS.positions).^2, 2));
    JAM_BS(j, nearestCH) = 1;
end

% Save data if needed
if flag_plot
    legend('Active UEs', 'Inactive UEs', 'Active Jammers', 'Inactive Jammers', 'UAV CHs');
    xlabel('x [m]');
    ylabel('y [m]');
    zlabel('z [m]');
end

% Save the positions to a .mat file
%save('pos_BS_UEs_JAMs.mat', 'UEs', 'BS', 'JAMs', 'UE_BS', 'JAM_BS');


end