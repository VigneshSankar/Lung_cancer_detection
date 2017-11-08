function [segmentedNodule,status] = returnSegmentedNodule(nodule,images_slice,window)
    
    
    segmentedNodule = zeros(size(images_slice),'uint16');
    % finding centre
    idx = find(nodule>0.99);
    [row,col] = ind2sub(size(nodule),idx);
    x = round(mean(row));
    y = round(mean(col));
    
    if x > 0
        % creating window

%         window = 96;
        nodule_mask = zeros(size(images_slice),'uint16');
        step=window/2;

        [xl,xh,yl,yh] = returnWindow(x,y,size(images_slice),step);
%         x-step:x+step-1,y-step:y+step-1
        % segmenting

        segmentedNodule = images_slice(xl:xh,yl:yh);
        status = true;
    else
        status = false;
    end
end

function [xl,xh,yl,yh] = returnWindow(x,y,size,step)

    if x - step > 0 && x + step < size(1)
        xl = x-step ; 
        xh = x+step-1 ; 
    elseif x - step <= 0
        xl = 1;
        xh = 96;
    elseif  x + step >= size(1)   
        xl = size(1)-step*2 ;
        xh = size(1) - 1;
    end
    
    if y - step > 0 && y + step < size(2)
        yl = y-step ; 
        yh = y+step-1 ; 
    elseif y - step <= 0
        yl = 1;
        yh = 96;
    elseif  y + step >= size(2)   
        yl = size(2)-step*2 ;
        yh = size(2) - 1;
    end
    

end