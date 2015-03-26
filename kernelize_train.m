function [ kernel ] = kernelize_train(std_data, h)
%KERNELIZE Kernelize data using Gaussian Kernel
%   
%  1. Calculate euclidean distance between data and itself and square it
%  2. Divide matrix by 2 h (tuning parameter)
%  3. Exp each element of (-1) * matrix

kernel = pdist2(std_data, std_data) .^ 2;
kernel = kernel / 2 * h;
kernel = exp(-1 * kernel);

end

