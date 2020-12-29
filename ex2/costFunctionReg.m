function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



h = sigmoid(X * theta);
 % Because the multiplication matrix size is 1x1, we don't need sum() anymore
left_equation =  (-y)' *log(h);
right_equation = (1-y)' * log(1-h);
summation =(left_equation - right_equation);
tempJ = (1/m) * summation;

% set theta(1) to zero because in regularized exclude the parameter theta(1) -> theta0 in theory
theta(1) = 0;

reg = lambda/(2*m) * sum(theta .^2);

J = tempJ + reg;

grad = (X'*(h - y))/m   + (lambda/m) * theta;

% =============================================================

end
