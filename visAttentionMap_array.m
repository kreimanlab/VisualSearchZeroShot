%% Author: Mengmi Zhang
%% Kreiman Lab
%% web: http://klab.tch.harvard.edu/
%% Date: April 5, 2018

load('results/result_30_31_2.mat');
x = mat2gray(x);
attentionmap = imresize(x,[224 224]);

stimuli = imread('sampleimg/array_2.jpg');
target = imread('sampleimg/target_2.jpg');
subplot(2,2,1);
imshow(target);
title('target');
subplot(2,2,2);
imshow(stimuli);
title('search image');
subplot(2,2,3);
imshow(attentionmap);
title('attention map');
subplot(2,2,4);
heat = heatmap_overlay(stimuli,attentionmap);
imshow(heat);
title('overlaid');