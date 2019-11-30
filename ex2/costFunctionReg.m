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

J = (-1/m) * sum(y.*log(h) + (1-y).*log(1-h));
reg_theta = theta;
reg_theta(1) = 0;
reg_J = (lambda / (2*m)) * (transpose(reg_theta) * reg_theta);
J = J + reg_J;

grad = (1/m) * transpose(X) * (h - y);
reg_grad = (lambda/m) * theta;
reg_grad(1) = 0;
grad = grad + reg_grad;

% =============================================================

end
