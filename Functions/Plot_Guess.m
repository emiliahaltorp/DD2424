% clear all; close all; clc
% 
[X, Y, y] = LoadBatch('Test10.mat');
% load('Best_Trained_Net.mat');


% data = A.data; 
% data = data(1:10,:)';
% y_tr = categorical(y_tr);
n = size(X,2);

[P, ~] = EvaluateClassifier(X_test, NetParams, ExMA);
[~, k] = max(P);
% y = categorical(y);
% k = categorical(k);

acc_test = ComputeAccuracy(X_test, y_test, NetParams, ExMA);
disp(['Accuracy test data: ' num2str(acc_test*100) '%']);

X = uint8(X);
I = reshape(X, 32,32,3,n);

figure
imshow(I(:,:,:,1))

im = 9;
figure(2)
for i = 1:im
    hold on
    subplot(1,im,i)
    imshow(I(:,:,:,i))
    title({['Guess: ' num2str(k(i))]
            ['True: ' num2str(y_test(i))]
            });
end
