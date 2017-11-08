function [st] = saveXMLData(slicenum,SOPInstanceUID,datasetPath,patient_gts_path,save_path_patient)
    d = dir([datasetPath,'*.xml' ]);
    if size(d,1)>1
        xml_file = [datasetPath , d(1).name];
    else
        xml_file = [datasetPath , d.name];
    end
    st = xml2struct(xml_file);
    
    
    SlicesMalValues = struct([]);
    s=[];
    numOfSlices = size(slicenum,2);
    for ns = 1: numOfSlices
        ReqSliceNum = slicenum{ns};
        ReqSliceID = SOPInstanceUID{ns};
        SlicesMalValues(ns).sliceNumber = slicenum{ns};
        SlicesMalValues(ns).SliceID = SOPInstanceUID{ns};

        numOfRad = size(st.LidcReadMessage.readingSession,2);

        for nr = 1:numOfRad
            rad1 = st.LidcReadMessage.readingSession{1,nr};
            mal = returnMalig(rad1,ReqSliceID);
            s(str2num(slicenum{ns}),nr) = mal;
            SlicesMalValues(ns).malOfRad(nr) = mal;
        end
        savedata = SlicesMalValues(ns);
%         save([save_path_patient,'slice',num2str(slicenum{ns}),'\','malignancy.mat'],'savedata')
    end
    csvwrite([save_path_patient,'malignancy.csv'],s)
    save([save_path_patient,'malignancy.mat'],'SlicesMalValues')
end