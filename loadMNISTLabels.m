function labels = loadMNISTLabels(filename)

fp = fopen(filename,'rb');
assert(fp ~= -1, ['could not open', filename, '']); %show an error message when it cannot open the file

magic = fread(fp, 1, 'bit 32', 0, 'ieee-be'); %Reading the data from the file we have opened earlier and loading data into a variable called magic
assert(magic == 2049, ['Bad magic number in', filename, '']); %prompt an error message during reading the file

numLabels = fread(fp, 1, 'bit 32', 0, 'ieee-be');
labels = fread(fp, inf, 'unsigned char');
assert(size(labels,1) == numLabels, 'Mismatch in label count'); %Prompt an error massege

fclose(fp);
end