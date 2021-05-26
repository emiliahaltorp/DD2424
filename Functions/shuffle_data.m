% Shuffle training examples randomly
%function [X_tr, X_val, X_test, Y_tr, Y_val, Y_test, y_tr, y_val, y_test] = shuffle_data(X_tr, X_val, X_test, Y_tr, Y_val, Y_test, y_tr, y_val, y_test)
clear all, close all, clc

[X_tr, Y_tr, y_tr]          = LoadBatch('Traindata80.mat');
[X_val, Y_val, y_val]       = LoadBatch('Verification10.mat');
[X_test, Y_test, y_test]    = LoadBatch('Test10.mat');

n1 = size(X_tr,2);
n2 = size(X_val,2);
n3 = size(X_test,2);

X = [X_tr X_val X_test];
Y = [Y_tr Y_val Y_test];
y = [y_tr y_val y_test];

n = size(X,2);

r = randperm(n);
X = X(:,r);
Y = Y(:,r);
y = y(r);

X_tr    = X(:,1:n1);
Y_tr    = Y(:,1:n1);
y_tr    = y(1:n1);

X_val   = X(:,n1+1:n1+n2);
Y_val   = Y(:,n1+1:n1+n2);
y_val   = y(n1+1:n1+n2);

X_test  = X(:,n1+n2+1:end);
Y_test  = Y(:,n1+n2+1:end);
y_test  = y(n1+n2+1:end);

data = X_tr;
labels = y_tr;
save('Traindata80.mat', 'data', 'labels');

data = X_val;
labels = y_val;
save('Verification10.mat', 'data', 'labels');

data = X_test;
labels = y_test;
save('Test10.mat', 'data', 'labels');

%end

