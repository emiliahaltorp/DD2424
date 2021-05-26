
function [NetParams, GDparams] = SGD_cyclic(grads, NetParams, GDparams)
    layers = length(NetParams.W);
    fn = fieldnames(grads);
    % Change the learning rate
    GDparams.t = GDparams.t+1;
    t_mod = mod(GDparams.t/GDparams.n_s,2);
    if 0 <= t_mod && t_mod <= 1 
        GDparams.n_t = GDparams.eta_min + t_mod * (GDparams.eta_max-GDparams.eta_min);
    else 
        GDparams.n_t = GDparams.eta_max - (t_mod-1) * (GDparams.eta_max-GDparams.eta_min);
    end
    for j = 1:length(fn) 
        for i = 1:layers
            NetParams.(fn{j}){i} = NetParams.(fn{j}){i} - GDparams.n_t * grads.(fn{j}){i};
        end
    end
end
