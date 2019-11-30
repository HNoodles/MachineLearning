function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% compute hidden layer 
X = [ones(m, 1) X];
all_a1 = arrayfun(@sigmoid, X * Theta1');

% compute the output layer
all_a1 = [ones(m, 1) all_a1];
all_a2 = arrayfun(@sigmoid, all_a1 * Theta2');

% get the index of max value
[~, p] = max(all_a2, [], 2);

% =========================================================================


end
