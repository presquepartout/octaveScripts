function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Initialize output variables: 
J = 0;
grad = zeros(size(theta));

% ====================== BEGIN ======================
% The neural network parameters Theta are computed by 
% logistic regression. 
%               J is the cost function. 
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. We can make use of this to vectorize
%       the cost function and gradient computations. 
%
% 

argumentSigmoid = X*theta;
g = sigmoid(argumentSigmoid);

% part1 is the component of the cost that is added when y=1

part1 = -y'*log(g);
utility_ones = ones(length(y), 1);

% part2 is the component of the cost that is added when y = 0
part2 = -(utility_ones - y)'*log(utility_ones - g);
theta_prime = theta;

% in the regularization term, we do not use constant coefficients.
theta_prime(1) = 0; 

J = ((part1 + part2) + lambda*(theta_prime'*theta_prime)/2)/m;
grad_part1 = g - y;
grad_part2 = X'*grad_part1;
grad = (grad_part2 + lambda.*theta_prime)/m;









% =============================================================

grad = grad(:);

end
