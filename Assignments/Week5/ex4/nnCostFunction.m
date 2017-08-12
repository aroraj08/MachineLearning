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


X = [ones(m, 1) X];
Z_2 = X * Theta1';
%m1 = size(Z_2, 1);
A_2 = sigmoid(Z_2);
A_2 = [ones(m, 1) A_2];
Z_3 = A_2 * Theta2';
h_theta_x = sigmoid(Z_3);

y_row = [1:num_labels];
for i = 1 : m 
	y_vec = (y_row == y(i))';
	sum = 0;
	for k = 1 : num_labels 
		sum = sum + (-y_vec(k) .* log(h_theta_x(i,k))) - ((1 - y_vec(k)).*(log(1 - h_theta_x(i,k))));
	end;
	J = J + sum;
end;


J = J/m;

%calculate reqhularized cost

Theta1_withoutBias = Theta1;
Theta1_withoutBias(:,1) = 0;

Theta2_WithoutBias = Theta2;
Theta2_WithoutBias(:, 1) = 0;

layer1_sum = 0;

for j = 1 : size(Theta1, 1); 
   for k = 1 : size(Theta1, 2); 
	layer1_sum =  layer1_sum + Theta1_withoutBias(j,k) .* Theta1_withoutBias(j,k);
   end 
end 

layer2_sum = 0; 
for j = 1 : size(Theta2, 1)
   for k = 1 :  size(Theta2, 2)
	layer2_sum = layer2_sum + Theta2_WithoutBias(j,k) .* Theta2_WithoutBias(j,k);
   end 
end 

regularizedCost  = (lambda * (layer1_sum + layer2_sum))/(2*m);

J = J + regularizedCost;

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


Theta2_withoutBias = Theta2(:,[2:end]);
%BigDelta2 = zeros(size(Theta2, 1),size(Theta2, 2)-1);
%BigDelta1 = zeros(size(Theta1, 1),size(Theta1, 2)-1);

BigDelta1 = zeros(size(Theta1));
BigDelta2 = zeros(size(Theta2));

for t = 1 : m
	
% Step 1

a_1 = X(t, :);
z_2 = Theta1 * a_1';
a_2 = sigmoid(z_2);
a_2_withBias = [1;a_2];
z_3 =  Theta2 * a_2_withBias;
a_3 = sigmoid(z_3);

% Step 2
 
delta_layer3 = zeros(size(Theta2, 1),1);

y_row = [1:num_labels];
y_vec = (y_row == y(t))';

for k = 1 : num_labels
   delta_layer3(k,1) = a_3(k,1) - y_vec(k);  			
end


% Step 3

delta_layer2 = (Theta2_withoutBias' * delta_layer3).* sigmoidGradient(z_2);

% Step 4

BigDelta1 = BigDelta1 + (delta_layer2 * a_1);
BigDelta2 = BigDelta2 + (delta_layer3 * a_2_withBias');


end

% Step 5


%display('size of BigDelta2'); display(size(BigDelta2));
%display('size of BigDelta1'); display(size(BigDelta1));

Theta1_grad =  BigDelta1/m;
Theta2_grad = BigDelta2/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

%display('grad is');  display(grad);
end
