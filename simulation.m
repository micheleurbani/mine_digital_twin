clear; close all; clc
% If a Python envoronment has not been specified, provide indications
if exist('pe', 'var')
    % Retrieve the path of the local folder
    localFolder = pwd;
    % Provide local python environment (important because required
    % libraries are installed there).
    pe = pyenv('Version',strcat(localFolder,'./venv/Scripts/python.exe'));
end

% Specify policy
shovelPolicy = [0.09489149715167379, 0.07855350468377378, 0.031012480624588046];
truckPolicy = [0.06383568139931146, 0.06256905803502141, 0.007347536683141531, 0.058981953297324585, 0.07878249800333016, 0.05058120249898333, 0.013447304537421411, 0.03475903556714466, 0.08335430697158006];
SIM_TIME = 10000;
SEED = 42;

param = struct();
param.shovelPolicy = shovelPolicy;
param.truckPolicy = truckPolicy;
param.simTime = SIM_TIME;
param.seed = SEED;
json_format = jsonencode(param);
fid = fopen('param.json', 'w');
if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, json_format, 'char');
fclose(fid);

% Run the simulation
output = sim(shovelPolicy,truckPolicy,SIM_TIME,SEED);
length(output.DumpSite0.StockpileHistory(:,2))
mean([output.DumpSite0.StockpileHistory(:,2); output.DumpSite1.StockpileHistory(:,2)])

function output = sim(shovelPolicy,truckPolicy,SIM_TIME,SEED)
%SIM function call a MatLab interpreter and execute a single simulation
%experiment using the parameters provided below. Function variables
%include:
%
%:param array shovelPolicy: an array with length from 1 to 4 where the
%thresholds to trigger preventive maintenance are stored.
%:param array truckPolicy: an array with length from 1 to 30 where the
%thresholds to trigger preventive maintenance are stored.
%:param int SIM_TIME: the length of the simulation horizon.
%:param int SEED: a value provided for reproducibility of the simulation
%instance.
%
%More parameters about the simulation experiment are provided to the Python
%interpreter by means of dictionary. Those parameters are the number of
%dumpsites and the number of workshops.
%Parameters which define the behaviour of trucks, shovels, dumpsites, and
%workshops are defined in the relative files within the folder "data".
%
%The function returns a strucs with the number of corrective and preventive
%interventions, and the stockpiles level histories.
    

    % Specify parameters
    param = py.dict(...
        pyargs(...
        'nTrucks', int32(length(truckPolicy)), ...      % Between 1 and 30
        'nShovels', int32(length(shovelPolicy)), ...    % Between 1 and 4
        'nDumpSites', int32(2), ...                     % Between 1 and 5
        'nWorkShops', int32(2), ...                     % Between 1 and 3
        'SIM_TIME', int32(SIM_TIME), ...                % One year expressed in minutes
        'SEED', int32(SEED), ...
        'thresholdsPM', py.dict(...
            pyargs(...
                'shovels', py.list(shovelPolicy), ...
                'trucks', py.list(truckPolicy) ...
            )... 
        )...
    ));
    
    output = py.main.std(param);
    
    % Conversion of data back to MatLab from Python dict
    output = struct(output);
    fNames = fieldnames(output);
    s = struct();
    
    for i = 1:size(fNames,1)
        % Separate procedures for trucks and shovels, and dumpsites
        if strcmp(fNames{i}(1:5),'Truck')

            field = struct(getfield(output,fNames{i}));

            tempFailureHistory = cell(field.FailureHistory);
            failureHistory = zeros(size(tempFailureHistory,1),3);
            if min(size(tempFailureHistory)) ~= 0
                for j = 1:size(tempFailureHistory,2)
                    failureHistory(j,:) = cellfun(@double,cell(tempFailureHistory{j}));
                end
            else
                failureHistory = zeros(1,3);
            end
            
            tempHistory = cell(field.History);
            History = cell(size(tempHistory,1),1);
            if min(size(tempHistory)) ~= 0
                for j = 1:size(tempHistory,2)
%                     History(j) = {[double(tempHistory{j}{1}), string(tempHistory{j}{2}), string(tempHistory{j}{3}), string(tempHistory{j}{4})]};
                    History{j,1} = double(tempHistory{j}{1});
                    History{j,2} = string(tempHistory{j}{2});
                    History{j,3} = string(tempHistory{j}{3});
                    History{j,4} = string(tempHistory{j}{4});
                end
            else
                History = zeros(1,4);
            end
            
            tempPreventiveMaintenanceHistory = cell(field.PreventiveMaintenanceHistory);
            preventiveMaintenanceHistory = zeros(size(tempPreventiveMaintenanceHistory,1),3);
            if min(size(tempPreventiveMaintenanceHistory)) ~= 0
                for j = 1:size(tempPreventiveMaintenanceHistory,2)
                    preventiveMaintenanceHistory(j,:) = cellfun(@double,cell(tempPreventiveMaintenanceHistory{j}));
                end
            else
                preventiveMaintenanceHistory = zeros(1,3);
            end
            temp = struct();
            temp.Failure = double(field.Failure);
            temp.FailureHistory = failureHistory;
            temp.PreventiveInterventions = double(field.PreventiveInterventions);
            temp.PreventiveMaintenanceHistory = preventiveMaintenanceHistory;
            temp.History = cell2table(History);
            temp.more_stats = struct(py.main.truckSummary(field.History));

            s = setfield(s,fNames{i},temp);
            
        elseif strcmp(fNames{i}(1:5), 'Shove')
            field = struct(getfield(output,fNames{i}));

            tempFailureHistory = cell(field.FailureHistory);
            failureHistory = zeros(size(tempFailureHistory,1),3);
            if min(size(tempFailureHistory)) ~= 0
                for j = 1:size(tempFailureHistory,2)
                    failureHistory(j,:) = cellfun(@double,cell(tempFailureHistory{j}));
                end
            else
                failureHistory = zeros(1,3);
            end
            
            tempPreventiveMaintenanceHistory = cell(field.PreventiveMaintenanceHistory);
            preventiveMaintenanceHistory = zeros(size(tempPreventiveMaintenanceHistory,1),3);
            if min(size(tempPreventiveMaintenanceHistory)) ~= 0
                for j = 1:size(tempPreventiveMaintenanceHistory,2)
                    preventiveMaintenanceHistory(j,:) = cellfun(@double,cell(tempPreventiveMaintenanceHistory{j}));
                end
            else
                preventiveMaintenanceHistory = zeros(1,3);
            end
            temp = struct();
            temp.Failure = double(field.Failure);
            temp.FailureHistory = failureHistory;
            temp.PreventiveInterventions = double(field.PreventiveInterventions);
            temp.PreventiveMaintenanceHistory = preventiveMaintenanceHistory;

            s = setfield(s,fNames{i},temp);
            
        elseif strcmp(fNames{i}(1:5),'DumpS')
            field = cell(getfield(output,fNames{i}));
            tempStockpileHistory = cell(field);
            stockpileHistory = zeros(size(field,2),2);
            if min(size(field)) ~= 0
                for j = 1:size(field,2)
                    stockpileHistory(j,:) = cellfun(@double,cell(tempStockpileHistory{j}));
                end
            else
                stockpileHistory = zeros(1,2);
            end
            temp = struct();
            temp.StockpileHistory = stockpileHistory;
            
            s = setfield(s,fNames{i},temp);
        end
    end
    
    output = s;
end

