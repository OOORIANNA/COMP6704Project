clear all, clc;

%% 数据读入
data1 = readmatrix("E:\Onedrive\OneDrive - email.swu.edu.cn\Nuts\1. Study\" + ...
    "1. SWU\2. 科研\2022.04.20 基于空气耦合超声的物体感测分类\2. 分类代码\dataset\" + ...
    "3. 训练总数据\chl1_10avr_55k-65k_nor_-1-1_pca95_rng1.csv");

data2 = readmatrix("E:\Onedrive\OneDrive - email.swu.edu.cn\Nuts\1. Study\" + ...
    "1. SWU\2. 科研\2022.04.20 基于空气耦合超声的物体感测分类\2. 分类代码\dataset\" + ...
    "3. 训练总数据\chl2_10avr_55k-65k_nor_-1-1_pca95_rng1.csv");

data3 = readmatrix("E:\Onedrive\OneDrive - email.swu.edu.cn\Nuts\1. Study\" + ...
    "1. SWU\2. 科研\2022.04.20 基于空气耦合超声的物体感测分类\2. 分类代码\dataset\" + ...
    "3. 训练总数据\chl3_10avr_55k-65k_nor_-1-1_pca95_rng1.csv");
data = [];
data(:,1,:) = data1;
data(:,2,:) = data2;
data(:,3,:) = data3;
labels = readmatrix("E:\Onedrive\OneDrive - email.swu.edu.cn\Nuts\1. Study\" + ...
    "1. SWU\2. 科研\2022.04.20 基于空气耦合超声的物体感测分类\2. 分类代码\dataset\" + ...
    "3. 训练总数据\labels_6k_rng1.csv");

% data = [data1, labels];
%% k折验证
 
[M,dim,N]=size(data); % M:总样本量； N:一个样本的元素总数
k_fold = 5;
indices=crossvalind('Kfold',data1(1:M,N),k_fold);  %进行随机分包
acc = [];
TrainInfo_all = [];  % 训练记录
for k=1:k_fold  %交叉验证k=10，10个包轮流作为测试集
    test = (indices == k);   %获得test集元素在数据集中对应的单元编号
    train = ~test;  %train集元素的编号为非test元素的编号
    XTrain = data(train,:,:);  %从数据集中划分出train样本的数据
    XTrain_new = cell(size(XTrain, 1), 1);
    for j = 1:size(XTrain_new,1)
        data_tmp = XTrain(j,:,:);
        data_tmp = reshape(data_tmp, [3,60]);
        XTrain_new{j} = data_tmp;
    end
    YTrain = categorical(labels(train,:));
    XTest = data(test,:,:);  %test样本集
    XTest_new = cell(size(XTest, 1), 1);
    for j = 1:size(XTest_new,1)
        data_tmp = XTest(j,:,:);
        data_tmp = reshape(data_tmp, [3,60]);
        XTest_new{j} = data_tmp;
    end
    YTest = categorical(labels(test,:));
    
    numObservations = numel(XTrain_new(:,1));
    sequenceLengths = [];
    for i=1:numObservations
        sequence = XTrain_new{i};
        sequenceLengths(i) = size(sequence,2);
    end
    [sequenceLengths,idx] = sort(sequenceLengths);
    XTrain_new = XTrain_new(idx,:);  % 报错地
    YTrain = YTrain(idx);

    %% train
    inputSize = 3;
    numHiddenUnits = 64;
    numClasses = 6;
    
    layers = [ ...
        sequenceInputLayer(inputSize)
        % convolution1dLayer(3, 40)
        bilstmLayer(numHiddenUnits,'OutputMode','last')
        bilstmLayer(numHiddenUnits,'OutputMode','last')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    maxEpochs = 2;
    batch_size = 16;
    
    options = trainingOptions('adam', ...
        'ExecutionEnvironment','cpu', ...
        'GradientThreshold',1, ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',batch_size, ...
        'SequenceLength','longest', ...
        'Shuffle','every-epoch', ...
        'Verbose',0, ...
        'Plots','training-progress');
   
    
    [net, TrainInfo] = trainNetwork(XTrain_new,YTrain,layers,options);
    TrainInfo_all = [TrainInfo_all, TrainInfo];
    %% test
    numObservationsTest = numel(XTest_new(:,1));
    sequenceLengthsTest = [];
    for i=1:numObservationsTest
        sequence = XTest_new{i};
        sequenceLengthsTest(i) = size(sequence,2);
    end
    
    [sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
    XTest_new = XTest_new(idx);
    YTest = YTest(idx);
    YPred = classify(net,XTest_new, ...
        'MiniBatchSize',batch_size, ...
        'SequenceLength','longest');
    acc_temp = sum(YPred == YTest)./numel(YTest);
    confusion_matrix1
    acc = [acc, acc_temp];
end

%%

