
% Computition of cost function for a set of images in X
function [J, L] = ComputeCost(X, Y, NetParams, lambda, varargin)
    % ---- Input ----
    % X     - Data matrix to normalize
    % Y     - One hot label
    % W     - Weight matrix
    % b     - Bias vector
    % lambda- 

    % ---- Output ----
    % J     - Cost matrix
    
    [P, ~] = EvaluateClassifier(X, NetParams, varargin{:});
    [row, col] = find(Y);
    lcross_sum = 0;
    for i = col'
        l_cross_i = -log(P(row(i),i));
        lcross_sum = lcross_sum + l_cross_i;
    end
    
    sum_W = 0;
    for i = 1:length(NetParams.W)
        sum_W = sum_W + sum(sum(NetParams.W{i}.^2));
    end
    
    L = lcross_sum/size(X,2);
    J = lcross_sum/size(X,2) + lambda * sum_W;
end
