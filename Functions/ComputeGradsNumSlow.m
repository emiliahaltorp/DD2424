
% Slow computation of gradients, numerical calculation
% (Function is given)
function Grads = ComputeGradsNumSlow(X, Y, NetParams, lambda, h)

Grads.W = cell(numel(NetParams.W), 1);
Grads.b = cell(numel(NetParams.b), 1);
if NetParams.use_bn
    Grads.gam = cell(numel(NetParams.gam), 1);
    Grads.beta = cell(numel(NetParams.beta), 1);
end

for j=1:length(NetParams.b)
    Grads.b{j} = zeros(size(NetParams.b{j}));
    NetTry = NetParams;
    for i=1:length(NetParams.b{j})
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) - h;
        NetTry.b = b_try;
        c1 = ComputeCost(X, Y, NetTry, lambda);        
        
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) + h;
        NetTry.b = b_try;        
        c2 = ComputeCost(X, Y, NetTry, lambda);
        
        Grads.b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(NetParams.W)
    Grads.W{j} = zeros(size(NetParams.W{j}));
        NetTry = NetParams;
    for i=1:numel(NetParams.W{j})
        
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) - h;
        NetTry.W = W_try;        
        c1 = ComputeCost(X, Y, NetTry, lambda);
    
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) + h;
        NetTry.W = W_try;        
        c2 = ComputeCost(X, Y, NetTry, lambda);
    
        Grads.W{j}(i) = (c2-c1) / (2*h);
    end
end

if NetParams.use_bn
    for j=1:length(NetParams.gam)
        Grads.gam{j} = zeros(size(NetParams.gam{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.gam{j})
            
            gam_try = NetParams.gam;
            gam_try{j}(i) = gam_try{j}(i) - h;
            NetTry.gam = gam_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda);
            
            gam_try = NetParams.gam;
            gam_try{j}(i) = gam_try{j}(i) + h;
            NetTry.gam = gam_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda);
            
            Grads.gam{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    for j=1:length(NetParams.beta)
        Grads.beta{j} = zeros(size(NetParams.beta{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.beta{j})
            
            beta_try = NetParams.beta;
            beta_try{j}(i) = beta_try{j}(i) - h;
            NetTry.beta = beta_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda);
            
            beta_try = NetParams.beta;
            beta_try{j}(i) = beta_try{j}(i) + h;
            NetTry.beta = beta_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda);
            
            Grads.beta{j}(i) = (c2-c1) / (2*h);
        end
    end    
end
end
    