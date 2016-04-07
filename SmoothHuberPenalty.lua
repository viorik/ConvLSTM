require 'nn'
require 'extracunn'

--[[
  This module does not modify its input, it only adds gradient in backprop to penalise for non-smoothness
  First we use a non-trainable convolutional layer, with fixed 5point stencil as filters, then we compute 
  L1 penalty on the result. 
--]]

SmoothHuberPenalty, parent = torch.class('nn.SmoothHuberPenalty', 'nn.Module')

function SmoothHuberPenalty:__init(transf, l1weight, threshold, sizeAverage)
  parent.__init(self)
  
  -- first layer is a non-trainable convolution with fixed 5point stencil as filters
  --local stencil = torch.Tensor(transf,3,3):zero():cuda()
  
  local gx = torch.Tensor(3,3):zero()
  gx[2][1] = -1/2
  gx[2][2] =  0
  gx[2][3] =  1/2
  gx = gx:cuda()
  local gradx = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
  gradx.weight:copy(gx)
  gradx.bias:fill(0)

  local gy = torch.Tensor(3,3):zero()
  gy[1][2] = -1/2
  gy[2][2] =  0
  gy[3][2] =  1/2
  gy = gy:cuda()
  local grady = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
  grady.weight:copy(gy)
  grady.bias:fill(0)
  
  local branchx = nn.Sequential()
  branchx:add(gradx):add(nn.Square())
  
  local branchy = nn.Sequential()
  branchy:add(grady):add(nn.Square())
  
  local gradconcat = nn.ConcatTable()
  gradconcat:add(branchx):add(branchy)
  
  local grad = nn.Sequential()
  grad:add(gradconcat)
  grad:add(nn.CAddTable())
  grad:add(nn.Sqrt())

  self.grad = grad
  --print (self.conv.weight)
  
  self.threshold = threshold or 0.001
  self.l1weight = l1weight or 0.01
  self.sizeAverage = sizeAverage or true      
end

function SmoothHuberPenalty:updateOutput(input)
  --self.output = input  
  self.output:resizeAs(input):copy(input)
  --self.output:renorm(2,1,0.33)
  return  self.output
end

function SmoothHuberPenalty:updateGradInput(input, gradOutput)
  local m = self.l1weight 
  if self.sizeAverage == true then 
    m = m/input:nElement()
  end
  self.gradInput:resizeAs(gradOutput)
  
  for i=1,input:size(1) do
    local dx = self.grad:updateOutput(input[{{i},{},{}}])
    dx:Huber(self.threshold)
    local gradL1 = dx:mul(m)
    self.gradInput[{{i},{},{}}] = self.grad:updateGradInput(input[{{i},{},{}}],gradL1):clone()
  end 
  
  --self.gradInput = self.conv:updateGradInput(input,gradL1) 
  self.gradInput:add(gradOutput)  
  --self.gradInput:resizeAs(gradOutput):copy(gradOutput) 
     
  return self.gradInput 
end
