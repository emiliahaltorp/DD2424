function [NetParams, GDparams, ExMA, tr, val] = TrainNet(X_tr, Y_tr, y_tr, X_val, Y_val, y_val, NetParams, GDparams, lambda, ExMA)

    % Itterate the number of times wanted
    for i = 1:GDparams.updates%(GDparams.cycles*2*GDparams.n_s)/GDparams.n_batch
        [NetParams, GDparams, ExMA] = MiniBatchGD(X_tr, Y_tr, GDparams, NetParams, ExMA, lambda);
    end
    
    [tr.cost, tr.loss] = ComputeCost(X_tr, Y_tr, NetParams, lambda);
    tr.acc = ComputeAccuracy(X_tr, y_tr, NetParams);

    [val.cost, val.loss] = ComputeCost(X_val, Y_val, NetParams, lambda);
    val.acc = ComputeAccuracy(X_val, y_val, NetParams);
    
end
    