function [V, D] = lda(X, labels)

% Make sure there's enough classes to do LDA (at least 2)
if length(unique(labels))<2
    error('Must have at least 2 classes to do LDA.')
end

% First, do PCA and project onto principal components with nonzero
% eigenvalues (this will guarantee that Sw is full rank)
[V_pca, D_pca, ~, ~, ~] = pca2(X);

% Use a tolerance of 1e-5 for consider an eigenvalue "zero"
%cut_pca = find(D_pca<=1e-5, 1)-1;
%V_pca = V_pca(:,1:cut_pca);

% Project data onto principal components with nonzero eigenvalues
X_pca = V_pca'*X;

% Get the dimension of the projected data
p = size(X_pca, 1);

% Get list of classes
classes = unique(labels);

% Compute the mean column of the data
mu = mean(X_pca, 2);

% Compute the between-class and within-class scatter matrices, Sb and Sw
Sw = zeros(p);
Sb = zeros(p);
for c=1:length(classes)
    % Get number of samples in class and an array of those samples
    n_c = sum(labels==classes(c));
    X_c = X_pca(:, labels==classes(c));

    % Compute class mean and add Sb and Sw components corresponding to this
    % class
    mu_c = mean(X_c, 2);
    Sb = Sb + n_c.*(mu_c-mu)*(mu_c-mu)';
    Sw = Sw + n_c*(X_c - mu_c*ones(1, n_c))*(X_c - mu_c*ones(1, n_c))';
end

% Find the eigenvector and eigenvalue matrices for LDA (i.e. the solution
% to Sb*V=Sw*V*D)
[V_lda, D] = eig(Sb, Sw);

% Re-sort eigenvalues and eigenvectors in descending order
[D, sort_ord] = sort(diag(D), 'descend');
V_lda = V_lda(:,sort_ord);

% Determine how many nonzero eigenvalues there will be (i.e. the minimum of
% rank(Sw) and rank(Sb)) and cut off irrelevant eigenvectors with
% eigenvalues of zero
cut_lda = min(rank(Sw), rank(Sb));

% Cut off irrelevant eigenvectors with eigenvalues of zero
V_lda = V_lda(:,1:cut_lda);
D = D(1:cut_lda);

% Combine the effects of PCA and LDA
V = V_pca*V_lda;

% Normalize the columns of V (unit eigenvectors)
V = normc(V);

end