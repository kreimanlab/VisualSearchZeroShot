%% Author: Mengmi Zhang
%% Kreiman Lab
%% web: http://klab.tch.harvard.edu/
%% Date: April 5, 2018

clear all;
close all;
clc;

% -- layers: numlayer, numtemplates, convsize
% -- layers: 5, 64, 14
% -- layers: 10, 128, 7
% -- layers: 17, 256, 4
% -- layers: 23, 512, 4
% -- layers: 24, 512, 2
% -- layers: 30, 512, 2
% -- layers: 31, 512, 1

FixData = [];
Fix_posx = {};
Fix_posy = {};

LayerList = [1];
indexlist = {0,0,0,0,0,0};

receptiveSize = 200; %for IOR and oracle checking
arraysize = 80; %maximum number of fixations;
w = 1024;
h = 1280;

NumImage = 1; %1 natural image only
scoremat = zeros(NumImage, arraysize);

for i = 1:NumImage
    
    trialname = 'gray004.jpg';
    
    path = ['sampleimg/gt' num2str(str2num(trialname(5:end-4))) '.jpg' ];
    gt = imread(path);
    gt = imresize(gt,[w,h]);
    gt = mat2gray(gt);
    gt = im2bw(gt,0.5);
    gt = double(gt);
    
    img = imread(['sampleimg/' trialname]);
    img = imresize(img, [w, h]);
    display(['img: ' num2str(i)]);
    
    piecedir = dir(['choppednaturaldesign/img_id' trialname(5:end-4)  '_*_layertopdown.mat']);
    wholeimg = zeros(length(LayerList),size(img,1), size(img,2));
   
    posx = [];
    posy = [];
    
    for l = 1: length(LayerList)
        for j = 1: length(piecedir)
            %j
            input = load(['choppednaturaldesign/' piecedir(j).name]);
            comp = imread(['choppednaturaldesign/' piecedir(j).name(1:end-17) '.jpg' ]);
            input = input.x;
            input = imresize(input, [size(comp,1) size(comp,2)]);
            C = strsplit(piecedir(j).name,'_');        
            startpos = str2num(C{3});
            endpos = str2num(C{4});
            wholeimg(l,startpos: startpos+size(comp,1) - 1, endpos: endpos+size(comp,2) - 1) = input;
            
        end
        wholeimg(l,:,:) = mat2gray(wholeimg(l,:,:));
    end
    
    wholeimg = squeeze(mean(wholeimg,1));
        
    %apply Inhibiton Of Return
    found = 0;
    fixtime = 1;
    salimg = wholeimg;
    for f = 1: arraysize

        [Y,idx] = max(salimg(:));
        [x y]= ind2sub(size(salimg),idx);
        
        posx = [posx; x];
        posy = [posy; y];
        
        fixatedPlace_leftx = x - receptiveSize/2 + 1;
        fixatedPlace_rightx = x + receptiveSize/2;
        fixatedPlace_lefty = y - receptiveSize/2 + 1;
        fixatedPlace_righty = y + receptiveSize/2;
        
        if fixatedPlace_leftx < 1
            fixatedPlace_leftx = 1;
        end
        if fixatedPlace_lefty < 1
            fixatedPlace_lefty = 1;
        end
        if fixatedPlace_rightx > size(gt,1)
            fixatedPlace_rightx = size(gt,1);
        end
        if fixatedPlace_righty > size(gt,2)
            fixatedPlace_righty = size(gt,2);
        end
        fixatedPlace = gt(fixatedPlace_leftx:fixatedPlace_rightx, fixatedPlace_lefty:fixatedPlace_righty);
        fixatedRGB = img(fixatedPlace_leftx:fixatedPlace_rightx, fixatedPlace_lefty:fixatedPlace_righty,:);
        
        %%%%%%%%%%%%%%% display the search process %%%%%%%%%%%%%%%
        displaysalimg = imresize( salimg, [480 640]);
        subplot(2,2,1);
        displaystimuli = imread(['sampleimg/' trialname ]);
        displaystimuli = imresize(displaystimuli, [480 640]);
        imshow(displaystimuli);
        title('stimuli');
        subplot(2,2,2);
        displaygt = imread(['sampleimg/gt' num2str(str2num(trialname(5:end-4))) '.jpg']);
        imshow(displaygt);
        title('target');
        subplot(2,2,3);
        displayheat = heatmap_overlay(displaystimuli, displaysalimg);        
        imshow(displayheat);
        title('attention map');
        hold on;
        plot(y/h*640,x/w*480,'ro','markers',20,'linewidth',2);
        hold off;        
        subplot(2,2,4);
        imshow(fixatedRGB);
        title('fixated area');
        
        drawnow;        
        pause(0.5);        
        %%%%%%%%%%%%%%% display the search process %%%%%%%%%%%%%%%
        
             
        if sum(sum(fixatedPlace)) > 0 
            found = 1;
            break;
        else
            found = 0;
                      
            fixtime = fixtime + 1;
            salimg(fixatedPlace_leftx:fixatedPlace_rightx, fixatedPlace_lefty:fixatedPlace_righty) = 0;
            salimg = mat2gray(salimg);
            
            
        end
        
    end
    
    Fix_posx = [Fix_posx posy'];
    Fix_posy = [Fix_posy posx'];
    display(['img id: '  num2str(i) '; target found at fixation step: ' num2str(fixtime) ]);
    if fixtime <= arraysize
        scoremat(i,fixtime) = 1;
    end
    
end

FixData.Fix_posx = Fix_posx;
FixData.Fix_posy = Fix_posy;

%% Comment this part to plot cummulative search performance
% plot(cumsum(mean(scoremat,1)))
% xlabel('fixation number');
% ylabel('cummulative performance');

