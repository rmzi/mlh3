function [ h, lambda ] = preprocess(data, labels, seed)
%PREPROCESS Prepares data for evaluation in hw3.m
%   Detailed explanation goes here

% Seed all random number generators with the same number
rng(seed);

% Number of examples
N = length(data(:,1));

% Standardize data
std_data = standardize(data);

% split data into training (80%) / holdout (20%)
rand_idxs = randperm(N)';

num_train = N * (0.8);
% num_hold_out = N - num_train;

train = std_data(rand_idxs(1:num_train),:);
train_labels = labels(rand_idxs(1:num_train),1);

hold_out = std_data(rand_idxs(num_train,:),:);
hold_out_labels = labels(rand_idxs(num_train,:),1);

% Get squared pair-wise euclidean distances between points in the training
% set
pw_dists = pdist2(train, train) .^ 2;

% get h's for testing by getting quantiles at various p's
hs = quantile(pw_dists(:), [0.1, 0.25, 0.5, 0.75, 0.9]);

% get lambdas [e^[0-10]]
lambdas = exp(linspace(0,10,11));

% Test each pair of (h, lambda) on model
errors = zeros(length(hs),length(lambdas));
i = 0;
j = 0;

for h = hs
    i = i + 1;
    for lambda = lambdas
        j = j + 1;
        errors(i,j) = evaluate_svm(train, train_labels, hold_out, hold_out_labels, h, lambda);
    end
end

[min_error,min_idx] = min(errors(:));

[h_idx, lambda_idx] = ind2sub(size(errors),min_idx);

h = hs(h_idx);
lambda = lambdas(lambda_idx);
 
disp(min_error)

end

