
function [NetParams, GDparams] = ADAM_update(grads, NetParams, GDparams)
    layers = length(NetParams.W);
    fn = fieldnames(grads);
    for j = 1:length(fn) 
        for i = 1:layers
            % Update weight matrix
            GDparams.m{j,i} = GDparams.b1 * GDparams.m{j,i} + (1-GDparams.b1)*grads.(fn{j}){i};
            GDparams.v{j,i} = GDparams.b2 * GDparams.v{j,i} + (1-GDparams.b2)*grads.(fn{j}){i}.*grads.(fn{j}){i};
            m_hat = GDparams.m{j,i}/(1- (GDparams.b1^GDparams.t));
            v_hat = GDparams.v{j,i}/(1- (GDparams.b2^GDparams.t));
%             W_add = GDparams.eta * m_hat .* (sqrt(v_hat)+ eps).^-1;
            W_add = GDparams.n_t * m_hat .* (sqrt(v_hat)+ eps).^-1;
            NetParams.(fn{j}){i} = NetParams.(fn{j}){i} - W_add;
        end
    end
end

