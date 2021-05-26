
% Function to initilize weight matrix and bias vectors
function NetParams = InitilizeParameters(layers, NetParams)
    for i = 1:length(layers)-1
%         std_W   = sqrt(1/layers(i)); % Xavier initialization
        std_W   = sqrt(2/layers(i)); % He initialization
%         std_W   = 1e-4;
        W{i}    = randn(layers(i+1),layers(i))*std_W;
        b{i}    = zeros(layers(i+1),1);
        gam{i}  = ones(layers(i+1),1);
        beta{i} = zeros(layers(i+1),1);
    end
    NetParams.W = W;
    NetParams.b = b;
    NetParams.gam = gam;
    NetParams.beta = beta;
end
