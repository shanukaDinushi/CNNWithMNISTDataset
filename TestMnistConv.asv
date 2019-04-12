Images = loadMNISTImages('MNIST/t10k-images.idx3-ubyte');
Images = reshape(Images, 28, 28, []);

Labels = loadMNISTLabels('MNIST/t10k-labels.idx1-ubyte');
Labels(Labels == 0) = 10;

randGen(1);

W1 = 1 - 2*randn([9 9 20]);
W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360+2000);
W0 = (2*rand(10, 100) - 1) * sqrt(6) / sqrt(10+100);

X = Images(:, :, 1:8000);
D = Labels(1:8000);

for epoch = 1:3
    epoch
    [W1, W5, W0] = MnistConv(W1, W5, W0, X, D);
end

save('MnistConv.mat');

X = Images(:, :, 1:8000);
D = Labels(1:8000);
acc = 0;
N = length(D);

for k = 1:N
    
        x = X(:, :, k);
        y1 = ConvalutionFunction(x, W1);
        y2 = ReLU(y1);
        y3 = Pool(y2);
        y4 = reshape(y3, [], 1);
        v5 = W5*y4;
        y5 = ReLU(v5);
        v = W0*y5;
        y = Softmax(v);
        
        [~, i] = max(y);
        if i == D(k)
            acc = acc+1;
        end
        
end

acc = acc / N;

fprintf('Accuracy is %f\n', acc);