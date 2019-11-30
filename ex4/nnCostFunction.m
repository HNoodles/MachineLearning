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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% feed forwarding
% compute hidden layer 
X = [ones(m, 1) X];
all_a2 = arrayfun(@sigmoid, X * Theta1');

% compute the output layer
all_a2 = [ones(m, 1) all_a2];
all_a3 = arrayfun(@sigmoid, all_a2 * Theta2');

h = all_a3;

% compute the cost
% first translate vector y into matrix in one-hot
temp = eye(num_labels);
expected_y = temp(y, :);
% then compute the cost part 
for i = 1:m
    J = J + (-1/m) * sum(expected_y(i, :).*log(h(i, :)) + (1-expected_y(i, :)).*log(1-h(i, :)));
end

% add regularization part, leave out the first column by convention
column_cost_theta1 = sum(Theta1.^2);
column_cost_theta2 = sum(Theta2.^2);
J = J + 1/2 * (lambda / m) * (sum(column_cost_theta1(:, 2:end), 2) + sum(column_cost_theta2(:, 2:end), 2));

for t = 1:m
    % input layer
    a1_t = X(t, :); % 1 * 401
    % hidden layer
    z2_t = a1_t * Theta1'; % 1 * 25
    a2_t = [1 sigmoid(z2_t)]; % 1 * 26
    % output layer
    z3_t = a2_t * Theta2'; % 1 * 10
    a3_t = sigmoid(z3_t); % 1 * 10
    
    % compute error
    delta3_t = a3_t - expected_y(t, :); % 1 * 10
    delta2_t = (Theta2' * delta3_t')' .* sigmoidGradient([1 z2_t]); % 1 * 26
    delta2_t = delta2_t(:, 2:end); % 1 * 25
    
    Theta2_grad = Theta2_grad + (1/m) * delta3_t' * a2_t;
    Theta1_grad = Theta1_grad + (1/m) * delta2_t' * a1_t;
end

Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

Theta2_grad = Theta2_grad + lambda / m * Theta2;
Theta1_grad = Theta1_grad + lambda / m * Theta1;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
