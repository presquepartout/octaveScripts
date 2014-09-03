function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%   gradientDescent performs these steps in the gradient descent algorithm:
%   - calculates the gradient of the cost function J
%   - saves the cost values for each iteration of gradient descent

%   X = matrix of training examples, m x n, n is number of features
%   m = number of training examples
%   y = vector of outcomes 
%   theta = n x 1 vector of parameter values (linear coefficients)
%   alpha = scalar multiplier of gradient vector (small alpha is 
%           fine-grained descent, large alpha uses bigger jumps)
%   num_iters = number of iterations of gradient descent that are performed
% 
%   theta = gradientDescent(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
scalar_number = alpha/m;

for iter = 1:num_iters

    % ====================== BEGIN ======================
    %  
    %
    %

     breakdown_h = X*theta;
     breakdownDiff = breakdown_h - y;
     gradient_vector = X'*breakdownDiff;
     correction = scalar_number.*gradient_vector;
     theta = theta - correction;
  
    % ============================================================

    % Save the cost J in every iteration  
 
    J_history(iter) = computeCost(X, y, theta);

end

end
