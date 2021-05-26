
function [NetParams, GDparams] = SGD_Momentum(grads, NetParams, GDparams)
    layers = length(NetParams.W);
    fn = fieldnames(grads);
    for j = 1:length(fn) 
        for i = 1:layers
%             GDparams.v_t{j,i} = GDparams.gam * GDparams.v_t{j,i} + GDparams.eta*grads.(fn{j}){i};
            GDparams.v_t{j,i} = GDparams.gam * GDparams.v_t{j,i} + GDparams.n_t*grads.(fn{j}){i};
            NetParams.(fn{j}){i} = NetParams.(fn{j}){i} - GDparams.v_t{j,i};
        end
    end
end

