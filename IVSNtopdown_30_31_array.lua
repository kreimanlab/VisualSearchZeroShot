-- Author: Mengmi Zhang
-- Kreiman Lab
-- web: http://klab.tch.harvard.edu/
-- Date: April 5, 2018

require 'torch'
require 'paths'
--require 'cudnn'
--require 'cunn'
require 'image'
require 'loadcaffe'
local matio = require 'matio'
tds = require 'tds'

local t = require './transforms'
local imagenetLabel = require './imagenet'

-- Load the caffe vgg16 model pre-trained on ImageNet
local cmodel = loadcaffe.load('Models/caffevgg16/VGG_ILSVRC_16_layers_deploy.prototxt', 'Models/caffevgg16/VGG_ILSVRC_16_layers.caffemodel', 'nn')

model_stimuli = nn.Sequential()
model_target = nn.Sequential()
-- layers: numlayer, numfeaturechannels, convolution kernel size
-- layers: 5, 64, 14
-- layers: 10, 128, 7
-- layers: 17, 256, 4
-- layers: 23, 512, 4
-- layers: 24, 512, 2
-- layers: 30, 512, 2
-- layers: 31, 512, 1


ConvSize = 1
NumTemplates = 512   
MyLayer = 31   
for i=1, 30 do --30        
	model_stimuli:add(cmodel:get(i))
end
print(model_stimuli)

for i=1,MyLayer do --31        
	model_target:add(cmodel:get(i))
end
print(model_target)

-- Evaluate mode
model_stimuli:evaluate()
model_target:evaluate()

MMconv = nn.SpatialConvolution(NumTemplates,1,ConvSize,ConvSize,1,1,1,1)

--module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])

-- The model was trained with this input normalization
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

TotalTrials = 600
targetsize = 28
stimulisize = 224

for i=2,2 do--TotalTrials do

   -- load the image as a RGB float tensor with values 0..1
   --imagename = string.format( "%08d", i )
   --Change this to your image
   imagename_stimuli = 'sampleimg/array_' .. i .. '.jpg'
   imagename_target = 'sampleimg/target_' .. i .. '.jpg'
   
   local stimuli = image.load(imagename_stimuli, 1, 'double')   
   stimuli = torch.cat({stimuli, stimuli, stimuli}, 1)   
   stimuli = image.scale(stimuli, stimulisize, stimulisize)
   stimuli = preprocess(stimuli)

   local target = image.load(imagename_target, 1, 'double')   
   target = torch.cat({target, target, target}, 1)   
   target = image.scale(target, targetsize, targetsize)
   target = preprocess(target)

   -- View as mini-batch of size 1
   local batch_stimuli = stimuli:view(1, table.unpack(stimuli:size():totable()))
   local batch_target = target:view(1, table.unpack(target:size():totable()))

   --feature maps from search network (will be convolved with output_target
   local output_stimuli = model_stimuli:forward(batch_stimuli):squeeze()  
   
   --feature maps from target network as convolution kernels 
   local output_target = model_target:forward(batch_target)

   

   print('target size')
   print(output_target:size())
   print('stimuli size')
   print(output_stimuli:size())
    
   --assign to convolution kernels
   MMconv.weight = output_target
    
   --convolve these two together
   out = MMconv:forward(output_stimuli:view(1, table.unpack(output_stimuli:size():totable()))):squeeze()
   print('attention map size')   
   print(out:size())

   print('trial#: ' .. i)
   savefile = 'results/result_30_31_' .. i .. '.mat'
   -- save output (attention map) to Matlab for visualization
   -- go to matlab and run MMvis.m
   matio.save(savefile,out)

end

print('done')
os.exit()
