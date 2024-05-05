% Function to initial locations of UEs and UAV CHs and Jammers
% Input:
%   noUsers   == double == number of active UEs
%   noSBS     == double == number of SBSs
%   flag_plot == bollean, = 1 to plot to test, = 0 to skip
function [UE_BS, JAM_BS, UEs, JAMs, BS] = location_voronoi_2D(num_cells, flag_plot)
% Output:
% UEs == 1x1 struct
%       UEs.active   == N_active x 2 matrix
%                               (1st col == x-coordinate
%                                2nd col == y-coordinate)
%       UEs.inactive == N_inactive x 2 matrix
%                               (1st col == x-coordinate
%                                2nd col == y-coordinate)
%       UEs.inBS     == 1 x N_active  : SBS that covers the active UEs
%                              example: UEs.inBS(2) = 4 means...
%                                         ...UE 2 in coverage of SBS 4
% BS  == 1x1 struct
%       BS.positions == N_sbs x 2 matrix
%                               (1st col == x-coordinate
%                                2nd col == y-coordinate)
%       BS.SBS       == N_sbs x 1 cell : save the positions of UEs that the SBS covers
%                       example: BS.SBS{1} == [150 100;
%                                              120 200;
%                                             -125 100]
%                                     --> SBS1 covers the UEs at (150,100),
%                                                     (120,200), (-125,100)
% UE_BS  == N x M matrix % matrix of relation of UEs and SBSs

% Initialization

%% Simulation window parameters
xMin=0;
xMax=10;
yMin=0;
yMax=100;

xDelta=xMax-xMin; %width 
yDelta=yMax-yMin; %height

%% Size of a square network 2D
network_size = xDelta*yDelta; %area of simulation window %1000; % meters

%% Active Users

lambda_U = 0.035;
num_users=poissrnd(network_size*lambda_U);%Poisson number of points
N_active_ue = num_users;
pos_users = network_size*rand([N_active_ue, 2]);

UEs.active(:,1) = pos_users(:,1);  % [m] x-coordinate of points
UEs.active(:,2) = pos_users(:,2);  % [m] y-coordinate of points

%% Inactive users

N_inactive_ue = 2;
pos_users_inactive = network_size*rand([N_inactive_ue, 2]);

UEs.inactive(:,1) = pos_users_inactive(:,1);  % [m] x-coordinate of points
UEs.inactive(:,2) = pos_users_inactive(:,2);  % [m] y-coordinate of points

%% Active jammers

lambda_active_ue = 10*1e-7;             % [point/m] mean density
intensity_active_ue = lambda_active_ue*network_size;    % average number of point inside circular region

lambda_J = 0.0065;
num_jammer=poissrnd(network_size*lambda_J);%Poisson number of points
N_active_jam = num_jammer;
pos_jammers = network_size*rand([N_active_jam, 2]);

JAMs.active(:,1) = pos_jammers(:,1);  % [m] x-coordinate of points
JAMs.active(:,2) = pos_jammers(:,2);  % [m] y-coordinate of points

%% Inactive jammers

N_inactive_jam = 1;
pos_jammers_inactive = network_size*rand([N_inactive_jam, 2]);

JAMs.inactive(:,1) = pos_jammers_inactive(:,1);  % [m] x-coordinate of points
JAMs.inactive(:,2) = pos_jammers_inactive(:,2);  % [m] y-coordinate of points

%% UAV CHs

lambda_sbs = 10*1e-6;             % [point/m] mean density
intensity_sbs = lambda_sbs*network_size;

N_sbs = num_cells;

pos_cluster_heads = randi(network_size, [N_sbs, 2]);
x_cluster_heads = pos_cluster_heads(:,1);
y_cluster_heads = pos_cluster_heads(:,2);

% Locations of SBSs
BS.positions(:,1) = x_cluster_heads;  % [m] x-coordinate of points
BS.positions(:,2) = y_cluster_heads;  % [m] y-coordinate of points


%% Plotting Location of Points Inside 2D Circular Region
if flag_plot
    figure;

    %plot(mbs_center(1), mbs_center(2),'md','MarkerFaceColor','m', 'HandleVisibility','off'); hold on;                % Location of points
    %text(mbs_center(:,1)+13, mbs_center(:,2), 'MBS', ...
    %    'HorizontalAlignment','left')

    plot(UEs.active(:,1), UEs.active(:,2), 'bo','MarkerFaceColor','b'); hold on;                % Location of points

    plot(UEs.inactive(:,1), UEs.inactive(:,2),'Mo','MarkerFaceColor','M'); hold on;                % Location of points
    
    plot(JAMs.active(:,1), JAMs.active(:,2),'ro','MarkerFaceColor','r'); hold on;

    plot(JAMs.inactive(:,1), JAMs.inactive(:,2),'Co','MarkerFaceColor','C'); hold on;

    plot(BS.positions(:,1), BS.positions(:,2),'gs','MarkerFaceColor','g'); hold on;                % Location of points

end
%% Voronoi network

%network_size = 2*Sqr.frameSize(1);
num_cells = N_sbs;
num_users = N_active_ue;

% Plot voronoi area of cells
if flag_plot
    voronoi(BS.positions(:,1), BS.positions(:,2),'b'); hold on

    % Assign labels to the points.
    nump = size(BS.positions,1);
    plabels = arrayfun(@(n) {sprintf('CH%d', n)}, (1:nump)');
    hold on
    Hpl = text(BS.positions(:,1)+13, BS.positions(:,2), plabels, ...
        'HorizontalAlignment','left');
end
%% Extract information of the network
% 1- Positions of cluster heads
% 2- How many nodes in each clusters (which is automatically assigned by
% voronoi function of Matlab)?
% 3- Positions of nodes in each cluster

cluster_info = cell(num_cells,1);
BS.SBS = cell(num_cells,1);
UE_BS  = zeros(size(UEs.active,1), size(BS.positions,1));
% matrix of relation of UEs and SBSs

for kk=1:num_users
    min_distance = network_size;
    cluster_id = 0;
    for uu=1:num_cells
        % distance between each user and its cluster head
        dist_user_ch = sqrt((UEs.active(kk,1)-BS.positions(uu,1))^2 ...
            + (UEs.active(kk,2)-BS.positions(uu,2))^2);
        if dist_user_ch < min_distance
            min_distance = dist_user_ch;
            cluster_id = uu;
        end
    end
    if cluster_id > 0
        BS.SBS{cluster_id} = [BS.SBS{cluster_id}; UEs.active(kk,1) UEs.active(kk,2) kk];
        UEs.inBS(kk) = cluster_id;
        UE_BS(kk,cluster_id) = 1;
    end
end

%%
JAM_BS = zeros(size(JAMs.active,1), size(BS.positions,1));
% matrix of relation of UEs and SBSs

for kk=1:num_jammer
    min_distance = network_size;
    cluster_id = 0;
    for uu=1:num_cells
        % distance between each jammer and its cluster head
        dist_jam_ch = sqrt((JAMs.active(kk,1)-BS.positions(uu,1))^2 ...
            + (JAMs.active(kk,2)-BS.positions(uu,2))^2);
        if dist_jam_ch < min_distance
            min_distance = dist_jam_ch;
            cluster_id = uu;
        end
    end
    if cluster_id > 0
        BS.SBS{cluster_id} = [BS.SBS{cluster_id}; JAMs.active(kk,1) JAMs.active(kk,2) kk];
        JAMs.inBS(kk) = cluster_id;
        JAM_BS(kk,cluster_id) = 1;
    end
end
%% Save positions of nodes in each cluster to separated files

% test
if flag_plot
    for jj = 1:num_cells
        if (~isempty(BS.SBS{jj}))
            text(BS.SBS{jj,1}(:,1)+13, ...
                BS.SBS{jj,1}(:,2), num2str(jj), ...
                'HorizontalAlignment','left'); hold on

            plot(BS.SBS{jj}(:,1), ...
                BS.SBS{jj}(:,2), ...
                '.', 'Color', 'None' ,'MarkerSize',15,'MarkerEdgeColor','none', 'HandleVisibility','off'); hold on
        end
    end

    legend('active Nodes', 'inactive Nodes', 'active jams', 'inactive jams', 'CHs');
    xlabel('$x$ [m]','Interpreter','LaTex');
    ylabel('$y$ [m]','Interpreter','LaTex');
    %saveas(gca,'Voronoi.png','png');
    save('pos_BS_UEs_JAMs_10Cells.mat', 'UEs', 'BS', "UE_BS", 'JAMs', 'JAM_BS')
end

end