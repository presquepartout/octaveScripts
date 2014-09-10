function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% initialize J and working matrices:  
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== BEGIN ======================
% The code has the following parts. 
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. Return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         

% Feed forward (calculate the output layer)
% Add ones to X

X = [ones(m, 1) X];

% Part 1: Feed Forward

% at each level of the neural net, 
% the z-variables are outputs of the theta-transformations;
% the a-variables are sigmoid transformations of the z-variables. 

z2 = X*Theta1';
a2 = sigmoid(z2);
A2 = [ones(m, 1) a2];
z3 = A2*Theta2';
a3 = sigmoid(z3);

% create a vector that will hold the logistic regression term
% of the cost

costvector = zeros(m, 1);

for i = 1:m,
    yvector = [1:num_labels]';
    yvector = yvector == y(i);
    part1 = -log(a3(i,:))*yvector;
    part2 = -log(1 .- a3(i,:))*(1 .- yvector);
    costvector(i) = part1 + part2;
    
end;
    
% compute regularization terms. 
% when computing regularization, we do not need the first row
% of the Theta parameter matrices. 

trimTheta1 = Theta1(:, [2:(input_layer_size + 1)]);
trimTheta2 = Theta2(:, [2:(hidden_layer_size + 1)]);
part3 = sum(sum(trimTheta1.*trimTheta1));
part4 = sum(sum(trimTheta2.*trimTheta2)); 


J = sum(costvector)/m + (part3 + part4)*lambda/2/m;


% Part 2: Backpropagation 

% compute gradients from backpropagation:
% initialize output values:

bigDelta_1 = zeros(size(Theta1));
bigDelta_2 = zeros(size(Theta2));
y_matrix = zeros(m, num_labels);

% populate y_matrix for vectorization: 
% y_matrix is a collection of the y-data in matrix form. 

for i = 1:m, 
    yvector = [1:num_labels];
    yvector = yvector == y(i);
    y_matrix(i, :) = yvector;
    
end;
    
    
  % compute 2nd level 
  % working backwards, first step:
  
delta_3 = a3 - y_matrix;

  % recall that A2 is the matrix a2 with row of ones added
   
bigDelta_2 = delta_3'*A2;
    
  % compute 1st level
  % working backwards, second step:
    
delta2_part1 = (Theta2')*delta_3';
workVector = delta2_part1(2:end, :);
    
delta_2 = workVector.*sigmoidGradient((z2)');
  
bigDelta_1 = delta_2*X;
    
  % compute gradients
    
Theta1_grad = bigDelta_1/m;

Theta2_grad = bigDelta_2/m;

% Part 3: Add Regularization terms


regTheta1 = zeros(size(Theta1));
regTheta1(:, 2:end) = Theta1(:,2:end);


regTheta2 = zeros(size(Theta2));
regTheta2(:, 2:end) = Theta2(:,2:end);



regTheta1_grad = Theta1_grad + lambda*regTheta1/m;
regTheta2_grad = Theta2_grad + lambda*regTheta2/m;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [regTheta1_grad(:) ; regTheta2_grad(:)];


end
