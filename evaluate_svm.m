function [ error ] = evaluate_svm(train, train_labels, hold_out, hold_out_labels, h, lambda)
%EVALUATE_SVM Use pair (h,lambda) and calculate error on hold-out set
%   Detailed explanation goes here

% Calculate kernel of training data and itself
train_kernel = kernelize_train(train, h);

% Get weights using ksvm trained on training data
alpha = hw3_train_ksvm(train_kernel, train_labels, lambda);

% Calculate kernel of training data with hold out data
test_kernel = kernelize_test(train, hold_out, h);

% Predict labels for hold_out set
preds = hw3_test_ksvm(alpha, test_kernel, train_labels);

% Calculate error of 
error = calc_error(hold_out_labels, preds);

end

