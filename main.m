%This code is a proof of concept for linear dimensionality reduction that 
%preserves the clustering structure, from the paper SqueezeFit: label-aware
%dimensionality reduction (McWirther, Mixon, Villar, arXiv:1812.02768).
%Assumes mnist data in local folder mnist_data. The experiment uses 4s and
%9s from MNIST (http://yann.lecun.com/exdb/mnist/) and evaluates the
%quality of the dimensionality reduction technique by looking at the
%k-nearest neighbors after projection.
%Requires cvx (http://cvxr.com/cvx/)

%parameters
loading=0; %requires to load the data
other_tests=1; %compares to other algorithms (LDA, PCA, no compression)
N=100; %number of training samples for squeezefit
K=3; %number of constraints per point
k=[1;5;15] %k-nearest neighbors classifier for different values of k
delta=0.2; %parameter regarding the prescribed margin (percentage of the smallest vector)

rng(1); %sets random seed for reproducibility
%filenames
file_data = 'mnist_data/train-images-idx3-ubyte';
file_labels = 'mnist_data/train-labels-idx1-ubyte';
file_test= 'mnist_data/t10k-images-idx3-ubyte';
file_test_labels= 'mnist_data/t10k-labels-idx1-ubyte';

addpath('utils');
%loads data (in this example just 4s and 9s from MNIST)
if loading==1
    [data,labels,test_data,test_labels]=load_mnist(file_data, file_labels, file_test, file_test_labels);
end


%randomly sample the data
n=size(data,2);
indices=randperm(n, N);
samples=data(:, indices);
samples_labels=labels(indices);

%sqz_performance
tic
[Delta,smallest]=select_constraints(samples, samples_labels, K);
M=sqz_sdp_hinge(Delta, delta*smallest, 1);
P=real(sqrt(M));
sqz_misclassification=nearest_neighbors_classifier(data, labels, P, test_data, test_labels, k)
toc


if other_tests==1
    
    %baseline_performance
    identity_misclassification=nearest_neighbors_classifier(data, labels, eye(100), test_data, test_labels, k)
    
    %pca k=3
    data_centered = bsxfun(@minus,data',mean(data'));
    [coeff,score,latent] =pca(data_centered);
    P=coeff(:,1:3)*coeff(:,1:3)';
    pca_3_misclassification=nearest_neighbors_classifier(data, labels, P, test_data, test_labels, k)
    
    %pca k=6
    data_centered = bsxfun(@minus,data',mean(data'));
    [coeff,score,latent] =pca(data_centered);
    P=coeff(:,1:11)*coeff(:,1:11)';
    pca_6_misclassification=nearest_neighbors_classifier(data, labels, P, test_data, test_labels, k)
    
    
    %lda in entire set
    [P_lda, ~] = lda(data, labels);
    lda_misclassification=nearest_neighbors_classifier(data, labels, P_lda', test_data, test_labels, k)
    
    %lda in samples
    [P_lda, ~] = lda(samples, samples_labels);
    lda_samples_misclassification=nearest_neighbors_classifier(data, labels, P_lda', test_data, test_labels, k)
end
