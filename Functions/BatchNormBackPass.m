function G = BatchNormBackPass(G, s, mu, v, n)
    sig1    = ((v+eps).^(-1/2));
    sig2    = ((v+eps).^(-3/2));
    G1      = G .* (sig1*ones(1,n));
    G2      = G .* (sig2*ones(1,n));
    D       = s - mu*ones(1,n);
    c       = (G2 .* D)*ones(n,1);
    G       = G1 - 1/n*(G1*ones(n,1))*ones(1,n) - 1/n*D .* (c*ones(1,n));
end