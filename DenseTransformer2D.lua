require 'nn'

local AGGOF, parent = torch.class('nn.AffineGridGeneratorOpticalFlow2D', 'nn.Module')

--[[
   AffineGridGeneratorOpticalFlow(height, width) :
   AffineGridGeneratorOpticalFlow:updateOutput(transformMap)
   AffineGridGeneratorOpticalFlow:updateGradInput(transformMap, gradGrids)

   AffineGridGeneratorOpticalFlow will take height x width x 2x3 an affine transform map (homogeneous 
   coordinates) as input, and output a grid, in normalized coordinates* that, once used
   with the Bilinear Sampler, will generate the next frame in the sequence according to the optical
   flow transform map.

   *: normalized coordinates [-1,1] correspond to the boundaries of the input image. 
]]

function AGGOF:__init(height, width)
   parent.__init(self)
   assert(height > 1)
   assert(width > 1)
   self.height = height
   self.width = width
   
   self.baseGrid = torch.Tensor(2, height, width)
   for i=1,self.height do
      self.baseGrid:select(1,1):select(1,i):fill(-1 + (i-1)/(self.height-1) * 2)
   end
   for j=1,self.width do
      self.baseGrid:select(1,2):select(2,j):fill(-1 + (j-1)/(self.width-1) * 2)
   end
   
   --self.baseGrid:select(1,3):fill(1)
end

function AGGOF:updateOutput(transformMap)
   assert(transformMap:nDimension()==3
          and transformMap:size(1)== 2
          , 'please input a valid transform map ')
   
   -- need to scale the transformMap
   
   self.output:resize(2, self.height, self.width):zero()
   self.output = torch.add(self.baseGrid,transformMap)
   
   return self.output
end

function AGGOF:updateGradInput(transformMap, gradGrid)
   self.gradInput:resizeAs(transformMap):zero()
   self.gradInput:copy(gradGrid)
   return self.gradInput
end
