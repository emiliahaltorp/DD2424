% Compute accuracy of the network predicitons
function acc = ComputeAccuracy_int(X, y, NetParams, varargin)
    % ---- Input ----
    % X     - Data matrix to normalize
    % y     - The label for each image
    % W     - Weight matrix
    % b     - Bias vector

    % ---- Output ----
    % acc   - Accuracy of network
    [P, ~] = EvaluateClassifier(X, NetParams,varargin{:});
    [~, k_star] = max(P);
    int1 = (k_star <= y+5);
    int2 = (k_star >= y-5);
    int = (int1 & int2);
    correct = sum(double(int));
    acc = correct / length(y);
end
