function [V, D, variance, X_centered, mu] = pca2(X)

% Get the size of the data
[p, n] = size(X);

% Get the mean feature vector
mu = mean(X, 2);

% Center the data
X_centered = X-mu*ones(1,n);

% Find eigendecomposition of covariance matrix (1/n)*XX' or (1/n)*X'X, whichever is
% smaller
if n<p
    [V, D] = eig(X_centered'*X_centered./n);
    % Map eigenvectors up to p-dimension
    V = X_centered*V;
    % Make sure columns still have magnitude of 1 (mapping up to
    % p-dimension may cause columns of V to no longer have norm 1)
    V = normc(V);
else
    % By default eigenvectors will have magnitude of 1
    [V, D] = eig(X_centered*X_centered'./n);
end

% Re-sort eigenvalues and eigenvectors in descending order
[D, sort_ord] = sort(diag(D), 'descend');
V = V(:,sort_ord);

% Compute the amount of variance preserved based on number of principal
% components retained
variance = cumsum(D)./sum(D);

end