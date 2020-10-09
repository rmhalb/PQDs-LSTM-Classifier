clc
clear all

%---- Load Database ---- %
%load('16PQDs_4800_NoNoise.mat')
load('16PQDs_4800_WithNoise.mat') % From the signal fenerator database creator

SignalsDataBaseCell = struct2cell(SignalsDataBase);

% signal %
Signals = SignalsDataBaseCell(2,:,:);
Signals = reshape(Signals, [1,length(Signals)])'; 

% labels %
Labels = SignalsDataBaseCell(1,:,:);
Labels = reshape(Labels, [1,length(Labels)]); 
Labels = categorical(Labels)';

% ---- Raw Signal Data for train and test --- %
[XTrain, YTrain, XTest, YTest] = train_test_prepare(Signals,Labels);

% ---- Define the BiLSTM Network Architecture --- %
layers = [ sequenceInputLayer(1)
           bilstmLayer(32,'OutputMode','last')
           tanhLayer('Name','tanh1')
           bilstmLayer(32,'OutputMode','last')
           tanhLayer('Name','tanh2')
           bilstmLayer(32,'OutputMode','last')
           tanhLayer('Name','tanh3')
           fullyConnectedLayer(16)
           softmaxLayer
           classificationLayer ]

% --- specify the training options for the classifier --- %
options = trainingOptions('adam', ...
    'MaxEpochs',63, ...
    'MiniBatchSize', 64, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'InitialLearnRate', 0.01, ...    
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);% , 'ValidationData',{XTest,YTest});

% ---  Train the BiLSTM Network --- %
[net, info] = trainNetwork(XTrain,YTrain,layers,options);
save('net_WithNoise.mat','net');      % Save - need to change the name in order to avoid overwrite
save('net_WithNoiseInfo.mat','info'); % Save - need to change the name in order to avoid overwrite


% --- Visualize the Training and Testing Accuracy --- %
trainPred = classify(net,XTrain);
LSTMAccuracyTrain = sum(trainPred == YTrain)/numel(YTrain)*100

testPred = classify(net,XTest);
LSTMAccuracyTest = sum(testPred == YTest)/numel(YTest)*100

cm = confusionchart(YTest,testPred);
cm.Title = 'Confusion Chart for BiLSTM';

