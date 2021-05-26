% Check Gradient computations
function Gradient_Check(X, Y, testParams, lambda)
    d_bound = 20; 
    n_bound = 10;
    testParams.W{1} = testParams.W{1}(:,1:d_bound);
    [P, BNParams] = EvaluateClassifier(X(1:d_bound,1:n_bound), testParams);
    grads_s = ComputeGradients(X(1:d_bound,1:n_bound), Y(:,1:n_bound), P, BNParams, testParams, lambda);
    grads = ComputeGradsNumSlow(X(1:d_bound, 1:n_bound), Y(:, 1:n_bound), testParams, lambda, 1e-5);
    fn1 = fieldnames(grads);
    fn2 = fieldnames(grads_s);
    % Absolute error
    fprintf('\nMaximum absolute error:\n')
    for i = 1:length(fn1)       
        check = AbsoluteError(grads_s.(fn2{i}), grads.(fn1{i}),fn1{i});
    end
    % Maximum relative error
    fprintf('\nMaximum relative error:\n')
    for i = 1:length(fn1)       
        check = RelativeError(grads_s.(fn2{i}), grads.(fn1{i}),fn1{i});
    end
end

        