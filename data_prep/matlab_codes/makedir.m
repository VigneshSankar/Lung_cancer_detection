function [c] = makedir(fullName);
    if exist(fullName)
        c = 'already exist';
    else
        mkdir(fullName);
        c = 'created new';
    end
end