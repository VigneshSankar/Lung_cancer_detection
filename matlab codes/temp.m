d = dir([datasetPath,'*.xml' ]);
xml_file = [datasetPath , d.name];
st = xml2struct(xml_file);

SlicesMalValues = struct([])


ReqSliceNum = slicenum{1}
ReqSliceID = SOPInstanceUID{1}
SlicesMalValues(1).sliceNumber = slicenum{1}
SlicesMalValues(1).SliceID = SOPInstanceUID{1}

numOfRad = size(st.LidcReadMessage.readingSession,2);

for nr = 1:numOfRad
    rad1 = st.LidcReadMessage.readingSession{1,nr}
    numofNodules = size(rad1.unblindedReadNodule,2)
    mal = returnMalig(rad1,ReqSliceID)
    SlicesMalValues(1).malOfRad(nr) = mal;
end