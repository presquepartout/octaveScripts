function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== BEGIN ======================
% This code sets the predicted values of the logistic 
% regression model. The probabilities returned by the 
% model are stored in h; the vector p(i) is zero if 
% h(i) < 0.5, one if h(i) >= 0.5.
%
h = zeros(m, 1);

h = sigmoid(X*theta);

p = h >= 0.5;




% =========================================================================


end
