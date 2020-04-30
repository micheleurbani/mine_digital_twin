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
param.shovelPolicy = [2734.068605783732, 2815.3895288778315, 3340.1960412906346];
% Declare truck policy
param.truckPolicy = [1122.0169782455648, 2795.403826898272, 627.217982556582, 2434.159259419229, 1835.745594687327, 1060.45888178791171, 662.4863635451953, 1010.80833154114735, 659.9638825046144, 2813.5131572589385];

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

output = cell(py.main.std(param));

experiment_results = jsondecode(string(output{1}));
items_status = jsondecode(string(output{2}));

% Update the initial time and change the seed (or left it blank)
param.initialTime = param.simTime;
param.seed = [];
% You can also change maintenance policies
param.shovelPolicy = [0.08, 0.06, 0.035];

% For the new run of the experiment the status of the items has to be
% provided
output = py.main.std(param, items_status);

