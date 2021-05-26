
% Function to calculate the relative error
function e = RelativeError(val1, val2, type)
    for i = 1:length(val1)
        e = max(max(abs(val1{i} - val2{i})./max(eps, abs(val1{i} + val2{i}))));
        disp(['Relative error grad_' type num2str(i) ': ' num2str(e)]);
    end
end
