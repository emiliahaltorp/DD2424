
% Perform mini batch gradient decent algorithm
function [NetParams, GDparams, ExMA] = MiniBatchGD(X, Y, GDparams, NetParams, ExMA, lambda)
    % Starting values
    n = size(X,2);
    n_batch = GDparams.n_batch;

    j = GDparams.j;
    epochs = GDparams.cycles*2*GDparams.n_s/GDparams.updates;
    % Generate mini batch
    for updates = 1:epochs%j=1:n/n_batch
        j = j+1;
        if j >= n/n_batch
            j = 1;
            % Shuffle training examples randomly
            r = randperm(n);
            X = X(:,r);
            Y = Y(:,r);
        end
        % Change the learning rate
        GDparams.t = GDparams.t+1;
        t_mod = mod(GDparams.t/GDparams.n_s,2);
        if 0 <= t_mod && t_mod <= 1 
            GDparams.n_t = GDparams.eta_min + t_mod * (GDparams.eta_max-GDparams.eta_min);
        else 
            GDparams.n_t = GDparams.eta_max - (t_mod-1) * (GDparams.eta_max-GDparams.eta_min);
        end
%         GDparams.tvec = [GDparams.tvec GDparams.n_t];
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = X(:, j_start:j_end); 
        Ybatch = Y(:, j_start:j_end);

        [P, BNParams] = EvaluateClassifier(Xbatch, NetParams);
        
        grads = ComputeGradients(Xbatch, Ybatch, P, BNParams, NetParams, lambda);
        % Update step
        [NetParams, GDparams] = ADAM_update(grads, NetParams, GDparams);
%         [NetParams, GDparams] = SGD_Momentum(grads, NetParams, GDparams);
%         [NetParams, GDparams] = SGD_cyclic(grads, NetParams, GDparams);
%         for i = 1:length(NetParams.W)
%             NetParams.W{i} = NetParams.W{i} - GDparams.n_t * grads.W{i};
%             NetParams.b{i} = NetParams.b{i} - GDparams.n_t * grads.b{i};
%             if NetParams.use_bn
%                 NetParams.gam{i} = NetParams.gam{i} - GDparams.n_t * grads.gam{i};
%                 NetParams.beta{i} = NetParams.beta{i} - GDparams.n_t * grads.beta{i};
%             end
%         end
        % Update exponetial moving average
        if NetParams.use_bn
            ExMA = Update_MovAvg(BNParams, ExMA);
        end
    end
    GDparams.j = j;
end
