
% Function to calculate absolute error
function e = AbsoluteError(val1, val2, type)
    for i = 1:length(val1)
        e = max(max(abs(val1{i} - val2{i})));
        disp(['Absolute error grad_' type num2str(i) ': ' num2str(e)]);
    end
end
