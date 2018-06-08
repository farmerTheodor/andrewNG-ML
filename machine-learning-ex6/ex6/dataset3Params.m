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
sigma = 0

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

testC = .2*(2 .^ [1:8]);
testSig = .02*(2 .^ [1:8]);
errorVal = zeros(8,8);
for i = 1:8
  testC(i)
  for j = 1:8
    testSig(j)
    model = svmTrain(X, y, testC(i), @(X, y) gaussianKernel(X, y, testSig(j)));
    predictions = svmPredict(model,Xval);
	  errorVal(i,j) = mean(double(predictions~=yval));
  endfor
endfor
minVal = min(min(errorVal));
[i, j] = find(errorVal == minVal);

C = testC(i)
sigma = testSig(j)
errorVal


% =========================================================================

end
