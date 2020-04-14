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
param.shovelPolicy = [0.09489149715167379, 0.07855350468377378, 0.031012480624588046];
% Declare truck policy
param.truckPolicy = [0.06383568139931146, 0.031012480624588046, 0.06256905803502141, 0.007347536683141531, 0.058981953297324585, 0.07878249800333016, 0.05058120249898333, 0.013447304537421411, 0.03475903556714466, 0.08335430697158006];

% Specify how many items are present in the mine
param.nShovels = length(param.shovelPolicy);
param.nTrucks = length(param.truckPolicy);
param.nDumpSites = 2;
param.nWorkShops = 2;

param.simTime = 100000;      % Length of thesimulation [minutes]
param.seed = 42;            % A value for the seed

% Encode the struct using JSON format
json_format = jsonencode(param);

% Unconmment the following code in case you need to save the parameters in
% JSON format to an external file
fid = fopen('param.json', 'w');
if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, json_format, 'char');
fclose(fid);

results = jsondecode(string(py.main.std(param)));

