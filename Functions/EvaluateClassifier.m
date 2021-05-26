
% Evaluation of the network function
function [P, BNParams] = EvaluateClassifier(X, NetParams, varargin)
    % ---- Input ----
    % X         - Data matrix
    % NetParams - Contains Weight matrix and Bias vector

    % ---- Output ----
    % P     - Probhability of each label
    
    % Loop through the layers to find the final probhability
    n = size(X,2);
    layers = length(NetParams.W);
    BNParams{1} = LayerScore(NetParams, X, 1, varargin{:});
    for i = 2:layers-1
        BNParams{i} = LayerScore(NetParams, BNParams{i-1}.X, i, varargin{:});
    end
    s = NetParams.W{layers}*BNParams{layers-1}.X + NetParams.b{layers}*ones(1,n);
    P = exp(s)./ sum(exp(s));
%     if isempty(varargin)
%         BNParams = Update_MovAvg(BNParams);
%     end
end
