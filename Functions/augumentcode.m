clear all; close all; clc

[X_tr, Y_tr, y_tr] = LoadBatch('Traindata80.mat');

% data = A.data; 
% data = data(1:10,:)';
X_tr = uint8(X_tr);
% y_tr = categorical(y_tr);
n = size(X_tr,2);
I = reshape(X_tr, 32,32,3,n);

figure(1)
montage(I(:,:,:,1:3), 'Size', [1,3])
I2 = flipdim(I ,2); 
figure(2)
montage(I2(:,:,:,1:3), 'Size', [1,3])
r = 10;%rand(1)*30;
I3 = imrotate(I,r,'crop');
I3 = uint8(I3);
figure(3)
montage(I3(:,:,:,1:3), 'Size', [1,3])
