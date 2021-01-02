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


% X = [ones(m,1) X];


% % foward propagation
% % a1 = X; 
% a2 = sigmoid(Theta1 * X');
% a2 = [ones(m,1) a2'];

% h_theta = sigmoid(Theta2 * a2'); % h_theta equals z3

% disp("h theta is");
% disp(size(h_theta));

% % y(k) - the great trick - we need to recode the labels as vectors containing only values 0 or 1 (page 5 of ex4.pdf)
% yk = zeros(num_labels, m); 
% for i=1:m,
%   yk(y(i),i)=1;
% end

% % follow the form
% J = (1/m) * sum ( sum (  (-yk) .* log(h_theta)  -  (1-yk) .* log(1-h_theta) ));



% % Note that you should not be regularizing the terms that correspond to the bias. 
% % For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix.
% t1 = Theta1(:,2:size(Theta1,2));
% t2 = Theta2(:,2:size(Theta2,2));

% % regularization formula
% Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);

% % cost function + reg
% J = J;


% -------------------------------------------------------------

% % Backprop

% for t=1:m,

% 	% dummie pass-by-pass
% 	% forward propag

% 	a1 = X(t,:); % X already have bias
% 	z2 = Theta1 * a1';

% 	a2 = sigmoid(z2);
% 	a2 = [1 ; a2]; % add bias

% 	z3 = Theta2 * a2;

% 	a3 = sigmoid(z3); % final activation layer a3 == h(theta)

	
% 	% back propag (god bless me)	
	
% 	z2=[1; z2]; % bias

% 	delta_3 = a3 - yk(:,t); % y(k) trick - getting columns of t element
% 	delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2);

% 	% skipping sigma2(0) 
% 	delta_2 = delta_2(2:end); 

% 	Theta2_grad = Theta2_grad + delta_3 * a2';
% 	Theta1_grad = Theta1_grad + delta_2 * a1; % I don't know why a1 doesn't need to be transpost (brute force try)

% end;

% % Theta1_grad = Theta1_grad ./ m;
% % Theta2_grad = Theta2_grad ./ m;


% % Regularization (here you go)


% 	Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
	
% 	Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));
	
	
% 	Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
	
% 	Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end));



a1 = [ones(size(X, 1), 1) X];

z2 =  a1 * Theta1';

temp =  sigmoid(z2);
a2 = [ones(size(temp,1) ,1 ) temp ];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

disp("a1 size is");
disp(size(a1));

disp("a2 size is");
disp(size(a2));

disp("a3 size is");
disp(size(a3));

disp("y is");
disp(size(y));

% [val,index] = max(a3,[],2);

% p = index;
  

y_matrix = eye(num_labels)(y,:) ;

disp("y matrix is");
disp(size(y_matrix));


h = a3;
left_equation = -(y_matrix) .* log(h);
disp("log h size is");
disp(size(log(h)));
right_equation = (1-y_matrix) .* log(1-h);

J= sum(sum(left_equation - right_equation))/m ;
% theta(1) = 0;

disp("theta 1 size  is");
disp(size(Theta1)); % 25 * 401

disp("theta 2 size is");
disp(size(Theta2)); % 10*26


%----------- Apply regularization -------------------

% Note that you should not be regularizing the terms that correspond to the bias. 
% For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix.
t1 = Theta1(:,2:size(Theta1,2)); 
disp("t1 size");
disp(size(t1)); 

t2 = Theta2(:,2:size(Theta2,2));
disp("t2 size ");
disp(size(t2)); % 10*25;


reg = lambda/(2*m) * (sum(sum(t1 .^ 2)) + sum(sum(t2 .^ 2)));

J = J + reg;


% disp("size J is");
% disp(size(J));


% J = tempJ + reg;

% grad = (X'*(h - y_matrix))/m   + (lambda/m) * theta

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
