function [slicenum,SOPInstanceUID,datasetPath] = returnTxtData(path)

    filename = path;
    delimiterIn = ' ';
    
    % headerlinesIn = 4;
    A = importdata(filename,delimiterIn);
    if(size(A,1)>3)
        datasetPath1 = strsplit(A{2},' ');
        datasetPath = datasetPath1{4};
        for i = 4:size(A,1)
            C = strsplit(A{i},' ');
            slicenum{i-3} = C{2};
            SOPInstanceUID{i-3} = C{9};
        end
    else
        slicenum{1}=false;
        SOPInstanceUID{1} = false;
        datasetPath{1} = false;
    end
end