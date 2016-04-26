require 'nn'
require 'cunn'
require 'DenseTransformer2D'
require 'SmoothHuberPenalty'

flow = nn.Sequential()
local pad = torch.floor(opt.kernelSizeFlow/2)
local conv = nn.Sequential()
conv:add(nn.SpatialConvolution(opt.nFiltersMemory[2],opt.transf,opt.kernelSizeFlow,opt.kernelSizeFlow,1,1,pad,pad))
local conv_new = require('weight-init')(conv, 'xavier')
flow:add(conv_new) -- regression layer 1

local conv2 = nn.Sequential()
conv2:add(nn.SpatialConvolution(opt.transf,opt.transf,opt.kernelSizeFlow,opt.kernelSizeFlow,1,1,pad,pad))
local conv_new2 = require('weight-init')(conv2, 'xavier')
flow:add(conv_new2) -- regression layer 2

-- add an extra convolutional layer useful to hard code initial flow map with 0
local conv0 = nn.SpatialConvolution(opt.transf,opt.transf,1,1,1,1)
conv0.weight:fill(0)
conv0.bias:fill(0)
flow:add(conv0)

-- need to rescale the optical flow vectors since the sampler considers the image size between [-1,1]
flow:add(nn.SplitTable(1))
local b1 = nn.Sequential()
local m1 = nn.Mul()
m1.weight = torch.Tensor{2/opt.memorySizeH}
b1:add(m1):add(nn.Reshape(1,opt.memorySizeH,opt.memorySizeW))
local b2 = nn.Sequential()
local m2 = nn.Mul()
m2.weight = torch.Tensor{2/opt.memorySizeW}
b2:add(m2):add(nn.Reshape(1,opt.memorySizeH,opt.memorySizeW))
local para = nn.ParallelTable()
para:add(b1):add(b2)
flow:add(para)
flow:add(nn.JoinTable(1))
-- clamp optical flow values to make sure they stay within image limits
flow:add(nn.Clamp(opt.dmin,opt.dmax))

-- next module does not modify its input, only accumulates penalty in backprop pass to penalise non-smoothness 
--flow:add(nn.SmoothHuberPenalty(opt.transf,opt.constrWeight[3]))

flow:add(nn.AffineGridGeneratorOpticalFlow2D(opt.memorySizeH,opt.memorySizeW)) -- apply transformations to obtain new grid

-- transpose to prepare grid in bhwd format for sampler
flow:add(nn.Transpose({1,3},{1,2}))
