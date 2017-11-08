
function [mal] = returnMalig(rad1,ReqSliceID)
    mal = 0;
    found = false;
    try
    numofNodules = size(rad1.unblindedReadNodule,2);
    catch
        mal=0;
        return
    end
    for nn = 1:numofNodules
        reqNodule=0;
        try
        nodule = rad1.unblindedReadNodule{1,nn};
        catch
            try
                nodule = rad1.unblindedReadNodule;
            catch
                mal=0;
                return
            end
        end
        numofSlices = size(nodule.roi,2);
        if numofSlices == 1
            sliceN = nodule.roi;
%             sliceN.imageSOP_UID.Text
            if sliceN.imageSOP_UID.Text == ReqSliceID
                reqNodule = nodule;
                try
                    mal = str2num(reqNodule.characteristics.malignancy.Text);
                catch
                    mal=0;
                end
                
            end
        else
            for ns = 1:numofSlices
                sliceN = nodule.roi{1,ns};
%                 sliceN.imageSOP_UID.Text
                if sliceN.imageSOP_UID.Text == ReqSliceID
                    reqNodule = nodule;
                    
                    try
                        mal = str2num(reqNodule.characteristics.malignancy.Text);
                        found = true;
                    catch
                        mal=0;
                    end
                    break
                end
                
            end
            
        end
        if found == true
            break;
        end
    end
end