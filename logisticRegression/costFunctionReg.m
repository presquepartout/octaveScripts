function [J, grad] = costFunctionReg(theta, X, y, lambda)
%costFunctionReg computes cost and gradient for logistic regression with regularization
%   J = costFunctionReg(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression 
%   grad = the gradient of the cost w.r.t. to the parameters. 
%
% Initialize some useful values
m = length(y); % number of training examples

% initialize return values:
J = 0;
grad = zeros(size(theta));

% ====================== BEGIN ======================
% The regularized cost function has an added quadratic
% term in the parameters theta. The strength of this 
% added term is governed by the scaling parameter 
% lambda. 
%               
%               


argumentSigmoid = X*theta;
g = sigmoid(argumentSigmoid);
part1 = -y'*log(g);
utility_ones = ones(length(y), 1);
part2 = -(utility_ones - y)'*log(utility_ones - g);
theta_prime = theta;
theta_prime(1) = 0;
J = ((part1 + part2) + lambda*(theta_prime'*theta_prime)/2)/m;
grad_part1 = g - y;
grad_part2 = X'*grad_part1;
grad = (grad_part2 + lambda.*theta_prime)/m;




% =============================================================

end
