clear; close all; clc
% If a Python envoronment has not been specified, provide indications
if exist('pe', 'var')
    % Retrieve the path of the local folder
    localFolder = pwd;
    % Provide local python environment (important because required
    % libraries are installed there).
    pe = pyenv('Version',strcat(localFolder,'./venv/Scripts/python.exe'));
end

% Create a struct variable where to store simulation parameters
param = struct();
% Declare shovel policy
param.shovelPolicy = [2734.068, 2815.389, 3340.196];

% Declare truck policy
param.truckPolicy = [1122.016, 2795.403, 627.217, 2434.159, 1835.745, 1060.458, 662.486, 1010.808, 659.963, 1813.513];

% Specify how many items are present in the mine
param.nShovels = length(param.shovelPolicy);
param.nTrucks = length(param.truckPolicy);
param.nDumpSites = 2;
param.nWorkShops = 2;

param.initialTime = 0;       % The initial time of the simulation [minutes]
param.simTime = 100000;      % Length of thesimulation [minutes]
param.seed = 42;             % A value for the seed
param.PMRule = "age_based";

% Encode the struct using JSON format
json_format = jsonencode(param);

% Unconmment the following code in case you need to save the parameters in
% JSON format to an external file
fid = fopen('param.json', 'w');
if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, json_format, 'char');
fclose(fid);

% We can optimize the number of truck and shovels that are required to
% reach a production target
production_target = 3*1e6;  % [kg]
n = 10;                     % test each configuration n times
result_tuple = cell(py.main.optimize_configuration(production_target, n, param));

% Change the number of trucks and shovels using the values found by the
% optimization procedure
param.truckPolicy = param.truckPolicy(1:double(result_tuple{1}));
param.shovelPolicy = param.shovelPolicy(1:double(result_tuple{2}));
param.nShovels = length(param.shovelPolicy);
param.nTrucks = length(param.truckPolicy);

% Execute the simulation experiment
output = cell(py.main.std(param));

experiment_results = jsondecode(string(output{1}));
items_status = jsondecode(string(output{2}));

% Update the initial time and change the seed (or left it blank)
param.initialTime = param.simTime;
param.seed = [];
% You can also change maintenance policies

% For the new run of the experiment the status of the items has to be
% provided
output = py.main.std(param, items_status);

