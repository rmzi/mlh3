function [ error ] = calc_error( labels, preds )
%CALC_ERROR Calculate the error of a classifier given labels and
%predictions
%   
%   err(labels, pred) = 1 / size(labels) * sum(preds(y) != labels(y)
%

% Find errors
mistake_rows = labels(:,1) ~= preds(:,1);

% Calculate error
error = length(labels(mistake_rows)) / length(labels(:,1));

end

