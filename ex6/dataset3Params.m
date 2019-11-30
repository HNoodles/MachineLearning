function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% list of values that can be used
list = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% matrix of errors
errors = zeros(size(list, 1), size(list, 1));

for i = 1:size(list, 1)
    for j = 1:size(list, 1)
        % decide which value to use in this iteration
        C = list(i);
        sigma = list(j);
        % train model
        model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
        % get prediction
        prediction = svmPredict(model, Xval);
        % compute error
        errors(i, j) = mean(double(prediction ~= yval));
    end
end

% find the least error index
[row, column] = find(errors == min(errors, [], "all"));

% set C and sigma
C = list(row);
sigma = list(column);

% =========================================================================

end
