function y = Pool(x)

[xrow, xcol, numFilters] = size(x);

y = zeros(xrow/2, xcol/2, numFilters);

for k = 1:numFilters
    filter = ones(2) / (2*2);
    image = conv2(x(:, :, k), filter, 'valid');
    y(:, :, k) = image(1:2:end, 1:2:end); %matlab has a predefined code in function conv2.We just have to call that function
end

end