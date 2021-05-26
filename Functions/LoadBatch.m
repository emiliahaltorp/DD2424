
% Reads data from the CIFAR-10 batch file
function [X, Y, y] = LoadBatch(filename)
    % ---- Input ----
    % filename - Name of CIFAR-10 batch file

    % ---- Output ----
    % X - Image pixel data          [dxn]
    % Y - One Hot label             [Kxn]
    % y - The label for each image  [1xn]
    
    A = load(filename);
    X = double(A.data);
    y = A.labels;
    
    %nr_lab = max(y);
    nr_lab = 117;
    Y = zeros(nr_lab, size(X,2));
    for i = 1:length(Y)
        Y(y(i),i) = 1;
    end
end
