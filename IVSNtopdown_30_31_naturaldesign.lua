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

-- Load the model
local cmodel = loadcaffe.load('Models/caffevgg16/VGG_ILSVRC_16_layers_deploy.prototxt', 'Models/caffevgg16/VGG_ILSVRC_16_layers.caffemodel', 'nn')
--print(cmodel)

model_stimuli = nn.Sequential()
model_target = nn.Sequential()
-- layers: numlayer, numtemplates, convsize
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
for i=1, 30 do       
	model_stimuli:add(cmodel:get(i))
end
--model_stimuli = model_stimuli:cuda()
print(model_stimuli)
-- Evaluate mode
model_stimuli:evaluate()

for i=1,MyLayer do      
	model_target:add(cmodel:get(i))
end
--model_target = model_target:cuda()
print(model_target)

-- Evaluate mode
model_stimuli:evaluate()
model_target:evaluate()

MMconv = nn.SpatialConvolution(NumTemplates,1,ConvSize,ConvSize,1,1,1,1)
--MMconv = nn.SpatialConvolution(NumTemplates,1,ConvSize,ConvSize,1,1,1,1):cuda()
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


ImageStrList = tds.Vec()
TotalNumImg = 0
TextFile = 'croppednaturaldesign_img.txt'
for line in io.lines(TextFile) do    
    TotalNumImg = TotalNumImg+1
    ImageStrList:insert(line)
end
print('total Num Img: ' .. TotalNumImg)

targetsize = 28
stimulisize = 224

target = image.load('sampleimg/targetgray004.jpg', 1, 'double')   
target = image.scale(target, targetsize, targetsize)
target = torch.cat({target, target, target}, 1) 
target = preprocess(target)


for i=1,TotalNumImg do

   -- load the image as a RGB float tensor with values 0..1
   imagename_stimuli = ImageStrList[i]
   
   local stimuli = image.load(imagename_stimuli, 1, 'double')   
   stimuli = torch.cat({stimuli, stimuli, stimuli}, 1)   
   stimuli = image.scale(stimuli, stimulisize, stimulisize)
   stimuli = preprocess(stimuli)

   -- View as mini-batch of size 1
   local batch_stimuli = stimuli:view(1, table.unpack(stimuli:size():totable()))
   local batch_target = target:view(1, table.unpack(target:size():totable()))

   -- Get the output of the softmax
   --local output_stimuli = model_stimuli:forward(batch_stimuli:cuda()):squeeze()  
   --local output_target = model_target:forward(batch_target:cuda())
   local output_stimuli = model_stimuli:forward(batch_stimuli):squeeze()  
   local output_target = model_target:forward(batch_target)
   
   MMconv.weight = output_target
   
   out = MMconv:forward(output_stimuli:view(1, table.unpack(output_stimuli:size():totable()))):squeeze()
   --print(out:size())

   print('trial#: ' .. i)
   savefile = string.sub(ImageStrList[i],1,-5) .. '_layertopdown.mat'
   matio.save(savefile,out)
   
end

os.exit()
