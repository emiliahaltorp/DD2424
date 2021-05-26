 % Update moving average
function ExMA = Update_MovAvg(BNParams, ExMA)
layers = length(BNParams);
    for i = 1:layers
        alpha = 0.9;
        if isfield(ExMA{i}, 'mu')
            ExMA{i}.mu  = alpha*ExMA{i}.mu + (1-alpha)* BNParams{i}.mu;
            ExMA{i}.v   = alpha*ExMA{i}.v + (1-alpha)* BNParams{i}.v; 
        else
            ExMA{i}.mu  = BNParams{i}.mu;
            ExMA{i}.v  = BNParams{i}.v;
        end
    end
end

