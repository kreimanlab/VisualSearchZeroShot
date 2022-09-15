clear all; clc; close all;
load('/media/mengmi/TOSHIBABlue1/Proj_VS/HumanExp/githuman/SubjectArray/array.mat');

trial = 8;
trial = MyData(trial);
targetcate = trial.targetcate;
targetnum = trial.targetnum;
iii = targetnum;
categ = targetcate;

%% Mengmi
%% Demo for the CocoApi (see CocoApi.m)
GenerateNum = 20000; %if img id is greater than 20000;  then use validation set to plot
FileDir = '/media/mengmi/TOSHIBA EXT/CategoryVisualSearch/Dataset/MSCOCO/';
% savefolder = 'train2014_OtherCategories';
dataType='val2014';
MODEDATA = 0; % 1 for train; 0 for val

if iii<20000
    dataType='train2014';
    
else
    dataType='val2014';
    iii = iii-20000;
end


%% initialize COCO api (please specify dataType/annType below)
annTypes = { 'instances', 'captions', 'person_keypoints' };
annType=annTypes{1}; % specify dataType/annType
annFile=sprintf([FileDir 'annotations/%s_%s.json'],annType,dataType);
coco=CocoApi(annFile);

%% display COCO categories and supercategories
if( ~strcmp(annType,'captions') )
  cats = coco.loadCats(coco.getCatIds());
  nms={cats.name}; fprintf('COCO categories: ');
  fprintf('%s, ',nms{:}); fprintf('\n');
  %nms=unique({cats.supercategory}); fprintf('COCO supercategories: ');
  %fprintf('%s, ',nms{:}); fprintf('\n');
end

%% get all images containing given categories, select one at random
 

display(nms{categ});
catIds = coco.getCatIds('catNms',{nms{categ}});
imgIds = coco.getImgIds('catIds',catIds);

imgId = imgIds(iii);
img = coco.loadImgs(imgId);
I = imread(sprintf([FileDir '%s/%s'],dataType,img.file_name));
%figure(1); imagesc(I); axis('image'); set(gca,'XTick',[],'YTick',[])
annIds = coco.getAnnIds('imgIds',imgId,'catIds',catIds,'iscrowd',[]);
anns = coco.loadAnns(annIds); coco.showAnns(anns);

S={anns.segmentation}; 

polygon = int32(S{1}{1});

%% binary mask
binarymask = zeros(size(I));
binarymask = insertShape(binarymask,'FilledPolygon',polygon,'Color', 'white','Opacity',1);
%imshow(binarymask);


%% object mask
objmask = binarymask.*double(I);
imshow(uint8(objmask));


 

