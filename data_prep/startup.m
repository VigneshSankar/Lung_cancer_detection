clear
clc

addpath(genpath('matlab_codes\LIDCToolbox-master'));
addpath(genpath('matlab_codes\xml2struct'));


% First use LIDC tool box to separate into images and masks
% LIDC_process_annotations

% use main to read XML files and slice corresponse to read the malignancy
% values for each slice
main