function [NetParams, GDparams, ExMA, tr, val] = TrainNet_rot(X_tr, Y_tr, y_tr, X_val, Y_val, y_val, NetParams, GDparams, lambda, ExMA)
    % Starting values to plot
    tr.acc  = zeros(1, 2*(GDparams.epochs+1));
    tr.cost = zeros(1, 2*(GDparams.epochs+1));
    tr.loss = zeros(1, 2*(GDparams.epochs+1));
    
    val.acc  = zeros(1, 2*(GDparams.epochs+1));
    val.cost = zeros(1, 2*(GDparams.epochs+1));
    val.loss = zeros(1, 2*(GDparams.epochs+1));
    
    [tr.cost(1), tr.loss(1)] = ComputeCost(X_tr, Y_tr, NetParams, lambda);
    tr.acc(1) = ComputeAccuracy(X_tr, y_tr, NetParams);

    [val.cost(1), val.loss(1)] = ComputeCost(X_val, Y_val, NetParams, lambda);
    val.acc(1) = ComputeAccuracy(X_val, y_val, NetParams);

    % Itterate the number of times wanted
    for i = 1:2:2*GDparams.epochs%GDparams.updates%(GDparams.cycles*2*GDparams.n_s)/GDparams.n_batch
        [NetParams, GDparams, ExMA] = MiniBatchGD(X_tr, Y_tr, GDparams, NetParams, ExMA, lambda);
        [tr.cost(i), tr.loss(i)] = ComputeCost(X_tr, Y_tr, NetParams, lambda);
        tr.acc(i) = ComputeAccuracy(X_tr, y_tr, NetParams);
        [val.cost(i), val.loss(i)] = ComputeCost(X_val, Y_val, NetParams, lambda);
        val.acc(i) = ComputeAccuracy(X_val, y_val, NetParams);
        X_rot = rotate_image(X_tr);
        [NetParams, GDparams, ExMA] = MiniBatchGD(X_rot, Y_tr, GDparams, NetParams, ExMA, lambda);
        [tr.cost(i+1), tr.loss(i+1)] = ComputeCost(X_tr, Y_tr, NetParams, lambda);
        tr.acc(i+1) = ComputeAccuracy(X_tr, y_tr, NetParams);
        [val.cost(i+1), val.loss(i+1)] = ComputeCost(X_val, Y_val, NetParams, lambda);
        val.acc(i+1) = ComputeAccuracy(X_val, y_val, NetParams);
    end
    
    [tr.cost(2*(GDparams.epochs+1)), tr.loss(2*(GDparams.epochs+1))] = ComputeCost(X_tr, Y_tr, NetParams, lambda);
    tr.acc(2*(GDparams.epochs+1)) = ComputeAccuracy(X_tr, y_tr, NetParams);

    [val.cost(2*(GDparams.epochs+1)), val.loss(2*(GDparams.epochs+1))] = ComputeCost(X_val, Y_val, NetParams, lambda);
    val.acc(2*(GDparams.epochs+1)) = ComputeAccuracy(X_val, y_val, NetParams);
    
end