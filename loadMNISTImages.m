function images = loadMNISTImages(filename)

fp = fopen(filename,'rb');
assert(fp ~= -1, ['could not open', filename, '']); %show an error message when it cannot open the file
magic = fread(fp, 1, 'bit 32', 0, 'ieee-be'); %Reading the data from the file we have opened earlier and loading data into a variable called magic
assert(magic == 2051, ['Bad magic number in', filename, '']); %prompt an error message during reading the file
numImage = fread(fp, 1, 'bit 32', 0, 'ieee-be');
numRows = fread(fp, 1, 'bit 32', 0, 'ieee-be');
numCols = fread(fp, 1, 'bit 32', 0, 'ieee-be');
images = fread(fp, inf, 'unsigned char=>unsigned char'); %reading the images
images = reshape(images, numCols, numRows, numImage); %reshape the images using numCols, numRows and numImage 
images = permute(images,[2 1 3]); %Ordering the images accordingly
fclose(fp);

images = reshape(images, size(images, 1)* size(images,2), size(images, 3));
images = double(images) / 255; %returns reshaped 8 bit images

end