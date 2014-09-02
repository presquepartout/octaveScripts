function J = computeCost(X, y, theta)

%computeCost computes the value of the cost function for linear regression. 
%   X = matrix of training examples. 
%   m = the number of rows of X: the number of training examples. 
%   n (not used) is the number of columns of X: the number of features. 
%   y = the m x 1 vector of outcomes. 
%   theta = n x 1 vector of linear coefficients for the linear model. 
%
%   J = computeCost(X, y, theta) is the cost of using theta as the
%   parameters for linear regression to fit the data points in X and y. 

% Initialize some useful values
m = length(y); % number of training examples

% Initialize J
J = 0;

% ====================== BEGIN ======================
% Compute the cost of a particular choice of theta. 
% J must be set to the final cost. 

sumDiffSquares = sum((X*theta - y).^2);
J = sumDiffSquares/2/m;



% =========================================================================

end
