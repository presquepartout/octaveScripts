function [X_norm, mu, sigma] = featureNormalize(X)
%featureNormalize normalizes the features in X 
%   featureNormalize(x) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% initialization
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== BEGIN ======================
% Note:         X is a matrix where each column is a 
%               feature and each row is an example. The 
%               normalization has to be performed for each
%               feature. 
%
% The following code works with the m x n matrix X. 
% The normalization of each column of X is: 
%  (original value - mean of column)/standard deviation
% This vectorized version computes a matrix of means, 
% and subtracts it from X. The vector sigma(x) stores 
% the standard deviations of each column. 

mu = mean(X);
sigma = std(X);
column_multiplier = ones(size(X, 1), 1);
mu_matrix = column_multiplier*mu;
mu_matrix(:,1) = 0;
sigma(1,1) = 1;
X_norm = (X - mu_matrix)./sigma;







% ============================================================

end
