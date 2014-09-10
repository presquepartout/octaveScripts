function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression
%   grad = gradient of the cost w.r.t. the parameters. 
%   

% Initialize some useful values
m = length(y); % number of training examples

% Initialize return values:  
J = 0;
grad = zeros(size(theta));

% ====================== BEGIN ======================

%
% Note: grad should have the same dimensions as theta
%

argumentSigmoid = X*theta;
g = sigmoid(argumentSigmoid);
part1 = -y'*log(g);
utility_ones = ones(length(y), 1);
part2 = -(utility_ones - y)'*log(utility_ones - g);
J = (part1 + part2)/m;
grad_part1 = g - y;
grad_part2 = X'*grad_part1;
grad = grad_part2/m;







% =============================================================

end
