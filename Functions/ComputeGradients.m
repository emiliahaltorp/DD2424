
% Compute gratients
function [grads] = ComputeGradients(X, Y, P, BNParams, NetParams, lambda)
    % ---- Input ----
    % X      - Data matrix 
    % Y      - The label for each image
    % P      - 
    % W      - Weight matrix
    % lambda -

    % ---- Output ----
    % grad_W - Gradient in weight matrix
    % grad_b - Gradient in bias vector
    
    n       = size(X,2);
    layers  = length(NetParams.W); 
    
    G       = -(Y-P);
    grad_W  = cell(layers,1);
    grad_b  = cell(layers,1);
    
    if NetParams.use_bn 
        grad_gam  = cell(layers,1);
        grad_beta  = cell(layers,1);
        
        grad_gam{layers}    = NetParams.gam{layers}*0;
        grad_beta{layers}   = NetParams.beta{layers}*0;

        for l = [layers:-1:2]
            
            
            grad_W{l}      = 1/n * G*BNParams{l-1}.X' + 2*lambda*NetParams.W{l};
            grad_b{l}      = 1/n * G*ones(n,1);
            G               = NetParams.W{l}'*G;
            G               = G.*((BNParams{l-1}.X)>0);

            grad_gam{l-1} = 1/n * (G .* BNParams{l-1}.s_hat)*ones(n,1);
            grad_beta{l-1}= 1/n * G*ones(n,1);

            G           = G.*(NetParams.gam{l-1}*ones(1,n));
            G           = BatchNormBackPass(G, BNParams{l-1}.s, BNParams{l-1}.mu, BNParams{l-1}.v, n);

        end
        grad_W{1}   = 1/n * G*X' + 2*lambda*NetParams.W{1};
        grad_b{1}   = 1/n * G*ones(n,1);
        
        grads.W     = grad_W;
        grads.b     = grad_b;
        grads.gam   = grad_gam;
        grads.beta  = grad_beta;
    else 
        for l = [layers:-1:2]
            grad_W{l}       = 1/n * G*BNParams{l-1}.X' + 2*lambda*NetParams.W{l};
            grad_b{l}       = 1/n * G*ones(n,1);
            
            G               = NetParams.W{l}'*G;
            G               = G.*((BNParams{l-1}.X)>0);
        end
        grad_W{1}       = 1/n * G*X' + 2*lambda*NetParams.W{1};
        grad_b{1}       = 1/n * G*ones(n,1);
        grads.W     = grad_W;
        grads.b     = grad_b;
    end
end
