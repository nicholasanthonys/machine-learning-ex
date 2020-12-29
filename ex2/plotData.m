function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Find indices of positive and negative examples
pos = find(y==1); 
neg = find(y==0);


% Plot examples
% X(pos,1) work as below :
% y = [ 2 3 1 4 0 1 2 6 0 4]
% X = [55 19;54 96;19 85;74 81;94 34;82 80;79 92;57 36;70 81;69 4]
% X(find(y==1), 1)
%ans =
%    19
%    82

plot(X(pos,1),X(pos,2), 'k+', 'LineWidth',2, 'MarkerSize',7);
plot(X(neg,1), X(neg,2), 'ko','MarkerFaceColor','y','MarkerSize',7);






% =========================================================================



hold off;

end
