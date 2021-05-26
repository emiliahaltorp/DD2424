
% Score each layer
function [BNParams, NetParams] = LayerScore(NetParams, X, i, varargin)
	n = size(X,2);
    s = NetParams.W{i}*X + NetParams.b{i}*ones(1,n);
    s_tilda = s;
    if NetParams.use_bn == 1
        if ~isempty(varargin)
            ExMA    = varargin{:}{i};
            mu      = ExMA.mu;
            v       = ExMA.v;
        else 
            mu              = mean(s,2);
            BNParams.mu     = mu;
%             v               = (s-mu*ones(1,n)).^2;
            v               = var(s,0,2);
            v               = v*(n-1)/n;
            BNParams.mu     = mu;
            BNParams.v      = v; 
        end
        % Normalize
        s_hat           = BatchNormalize(s, mu, v);
        BNParams.s_hat  = s_hat;
        s_tilda         = NetParams.gam{i} .* s_hat + NetParams.beta{i};
    end
    BNParams.X = max(0, s_tilda);
    BNParams.s = s;
end

