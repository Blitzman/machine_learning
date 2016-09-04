function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
num_examples_ = length(y); % number of training examples
num_weights_ = length(theta);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    theta_new_ = theta;
    
    for t=1:num_weights_
        h = 0;
        for i=1:num_examples_
            h = h + (hypothesis(theta, X(i,:)') - y(i)) * X(i,t);
        end
        theta_new_(t) = theta(t) - alpha / num_examples_  * h;
    end
    
    theta = theta_new_;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
