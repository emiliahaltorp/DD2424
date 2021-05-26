
% Normalize data in X according to the training data mean
function [X] = NormalizeData(X, mean, std)
    % ---- Input ----
    % X     - Data matrix to normalize
    % mean  - Mean of training data
    % std   - Standard deviation of training data

    % ---- Output ----
    % X     - Normalized data matrix
    
    X = X - repmat(mean, [1, size(X, 2)]); 
    X = X ./ repmat(std, [1, size(X, 2)]);
end
