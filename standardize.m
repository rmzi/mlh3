function [ standardized_data ] = standardize(data)
%STANDARDIZE Standardize data using mean and std
%   
%  1. Calculate mean of all features
%  2. Calculate std of all features
%  3. Subtract mean and divide by std dev on data

% Number of examples
N = length(data(:,1));

% Number of features
M = length(data(1,:));

% Means of each feature
feat_means = mean(data);

% Standard Deviation of each feature
feat_std = std(data);

% Subtract respective mean from each column
sub_data = data - repmat(feat_means, N, 1);

% Divide each column by respective std
standardized_data = sub_data ./ repmat(feat_std, N, 1);

end

