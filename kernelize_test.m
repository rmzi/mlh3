function [ kernel ] = kernelize_test(std_train, std_test, h)
%KERNELIZE Kernelize data using Gaussian Kernel
%   
%  1. Calculate euclidean distance between test and training and square it
%  2. Divide matrix by 2 h (tuning parameter)
%  3. Exp each element of (-1) * matrix

kernel = pdist2(std_test, std_train) .^ 2;
kernel = kernel / 2 * h;
kernel = exp(-1 * kernel);

end

