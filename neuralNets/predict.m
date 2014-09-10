function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Initialize output values: 

p = zeros(size(X, 1), 1);
% add ones to the X matrix: 
X = [ones(m, 1) X];



% ====================== BEGIN ======================
% The following code makes predictions using
%               a learned neural network. The variable p is a 
%               vector containing labels between 1 to num_labels.
%
% The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

z1 = Theta1*X'; % the first affine transformation in the neural network
z2 = sigmoid(z1); % the first transformed layer in the neural network

mz2 = size(z2,2); 

z2 = [ones(1, mz2); z2]; % for the next transformation, add row of ones to z2

z3 = Theta2*z2;
A = sigmoid(z3); % this is the final output of the neural network

% A is a matrix of probabilities of each class. 

[a, b] = max(A); % the max function returns indexes of maximum values. 

% p is set to the index of the max probability, which corresponds to
% the value of the label (e.g. index of 5 corresponds to max probability
% that the class is 5). 

p = b;







% =========================================================================


end
