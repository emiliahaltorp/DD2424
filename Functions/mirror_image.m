function X = mirror_image(X)
    n   = size(X,2);
    I   = reshape(X, 32,32,3,n);
    I   = flipdim(I ,2);
    X   = reshape(I, 32*32*3, n);
end