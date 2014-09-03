function [theta] = normalEqn(X, y)
%normalEqn computes the closed-form solution to linear regression 
%   normalEqn(X,y) computes the closed-form solution to linear 
%   regression using the normal equations derived from calculus. 

theta = zeros(size(X, 2), 1); %initialize theta to number of columns of X

% ====================== BEGIN ======================
% 


square = X'*X;
inv_square = pinv(square);
X_transform_y = X'*y;
theta = inv_square*X_transform_y;

% -------------------------------------------------------------


% ============================================================

end
