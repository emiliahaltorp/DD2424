
% Compute accuracy of the network predicitons
function acc = ComputeAccuracy(X, y, NetParams, varargin)
    % ---- Input ----
    % X     - Data matrix to normalize
    % y     - The label for each image
    % W     - Weight matrix
    % b     - Bias vector

    % ---- Output ----
    % acc   - Accuracy of network
    [P, ~] = EvaluateClassifier(X, NetParams,varargin{:});
    [~, k_star] = max(P);
    acc = sum(k_star==y) / length(y);
end
