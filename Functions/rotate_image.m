function X = rotate_image(X)
    n   = size(X,2);
    I   = reshape(X, 32,32,3,n);
    r   = 10;%rand(1)*30;
    I   = imrotate(I,r,'crop');
    X   = reshape(I, 32*32*3, n);
end