%%%%%%%%
% - Output of the LIDC toolbox is processed.
% - The mask for each nodules is applied on the CT scan and 96x96 window is
% used around the nodule to crop the data.
% - Using the information from the slice_correspondences, The XML file for
% each patient is found and malignancy value is extracted
%
%%%%%%%%
clear
clc

% Set these paths correctly...
LIDC_path   = 'D:\Projects\LIDC\';  % REPLACE WITH LIDC DATSET PATH
output_path = 'D:\Projects\Output_toolbox1\';  % REPLACE WITH OUTPUT PATH
save_path = 'D:\Projects\segmentedNodules1\';
makedir(save_path);

images_path = [output_path,'images\'];
images_names = dir(images_path);

% going through the output of LIDC to segment the nodules and extract XML
% data
for i = 1:size(images_names,1)
    display(['patient num = ' , num2str(i), ' total patietns = ', num2str(size(images_names,1))])
    patient_name = images_names(i).name;

    save_path_patient = [save_path,patient_name,'\'];
    makedir(save_path_patient);

    patient_image_path = [output_path,'images\',patient_name,'\'];
    patient_gts_path =  [output_path,'gts\',patient_name,'\'];

    patient_slices = dir([patient_image_path,'slice*.tif']);

     %% read slice_correspondence information information
     txt_path = [patient_gts_path,'slice_correspondences.txt'];
     [slicenum,SOPInstanceUID,datasetPath] = returnTxtData(txt_path);
     if slicenum{1} ~= false
         
         


        for sn = 1:size(patient_slices,1)
            images_slice = imread([patient_image_path,patient_slices(sn).name]);
    %         imagesc(images_slice)


            %% slice
            patient_slices_fullname = strsplit(patient_slices(sn).name,'.');
            patient_slices_name = patient_slices_fullname{1};
            gts_slices_path  = [output_path,'gts\',patient_name,'\',patient_slices_name,'\'];


            save_path_patient2 = [save_path_patient,'\',patient_slices_name,'\'];
            makedir(save_path_patient2);




            %% nodules
            gts_slice_nodules = dir([gts_slices_path,'GT_id*.tif']);
            k=0;
            segmentedNodule = {};
            for nn = 1:size(gts_slice_nodules,1)    
                nodule = imread([gts_slices_path,gts_slice_nodules(nn).name]);
                window = 96;
                [segmentedNoduleResult,status] = returnSegmentedNodule(nodule,images_slice,window);

                if status
                    k=k+1;
                    segmentedNodule{k} = segmentedNoduleResult;
                    saveImages(segmentedNodule{k},[save_path_patient2,'nodule',num2str(nn)])
                    imwrite(segmentedNodule{k},[save_path_patient2,'nodule_image',num2str(nn),'.tiff']);
                    imwrite(nodule,[save_path_patient2,'nodulemask',num2str(nn),'.tiff']);
                end
            end
            
                save([save_path_patient2,'segmentednodule.mat'],'segmentedNodule');
            
        end
        %%save XML for the corresponding slicenum, SOPInstanceUID
        if size(patient_slices,1)>0
            [a] = saveXMLData(slicenum,SOPInstanceUID,datasetPath,patient_gts_path,save_path_patient);
        end
    %     imagesc
    end
end

function [] = saveImages(image,path)
    imagesc(image)
    set(gca,'XTick',[]) % Remove the ticks in the x axis!
    set(gca,'YTick',[]) % Remove the ticks in the y axis
    set(gca,'Position',[0 0 1 1]) % Make the axes occupy the hole figure
    saveas(gcf,path,'png')
end