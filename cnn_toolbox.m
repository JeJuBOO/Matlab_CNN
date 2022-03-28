%2021.05.19
% CNN used Matlab toolbox
%% Setup
clear;
load mnist/trainingData
images = reshape(images, [28,28,1,60000]);
labels = categorical(labels);

%% train
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'Verbose',false, ...
    'Plots','training-progress');

trainedNet = trainNetwork(images,labels,layers,options);
save('matlab1', 'trainedNet');

%% test
load MNIST/testingData
images = reshape(images, [28,28,1,size(images,3)]);
labels = categorical(labels');

tic
[guesses, scores] = classify(trainedNet, images);
toc
fprintf('accuracy: %.2f%%\n',sum(guesses==labels)/numel(labels)*100);

%% visualize
figure(1);clf;
for im = 1:5
    t = activations(trainedNet, images(:,:,:,im), 5);
    for i = 1:size(t,3)
        subplot(5,size(t,3), (i-1) + (im-1)*size(t,3) + 1);
        image(t(:,:,i)*20);
        colormap gray
    end
end
figure(2);clf;
for im = 1:5
    t = activations(trainedNet, images(:,:,:,im), 9);
    for i = 1:size(t,3)
        subplot(5,size(t,3), (i-1) + (im-1)*size(t,3) + 1);
        image(t(:,:,i)*20);
        colormap gray
    end
end
figure(3);clf;
for im = 1:5
    t = activations(trainedNet, images(:,:,:,im), 12);
    for i = 1:size(t,3)
        subplot(5,size(t,3), (i-1) + (im-1)*size(t,3) + 1);
        image(t(:,:,i)*20);
        colormap gray
    end
end