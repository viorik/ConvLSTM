require 'nn'
require 'rnn'
require 'UntiedConvLSTM'
require 'DenseTransformer2D'
require 'SmoothHuberPenalty'
require 'encoder'
require 'decoder'
require 'flow'
require 'stn'

model = nn.Sequential()

-- add encoder
local seqe = nn.Sequencer(encoder)
--seqe:remember('both')
seqe:training()
model:add(seqe)

-- memory branch
local memory_branch = nn.Sequential()
--local seq = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[1],opt.nFiltersMemory[2], opt.nSeq, opt.kernelSize, opt.kernelSizeMemory, opt.stride))
local seq = nn.Sequencer(nn.UntiedConvLSTM(opt.nFiltersMemory[1],opt.nFiltersMemory[2], opt.nSeq, opt.kernelSize, opt.kernelSizeMemory, opt.stride))
seq:remember('both')
seq:training()
memory_branch:add(seq)
memory_branch:add(nn.SelectTable(opt.nSeq))
memory_branch:add(flow)

-- keep last frame to apply optical flow on
local branch_up = nn.Sequential()
branch_up:add(nn.SelectTable(opt.nSeq))

-- transpose feature map for the sampler 
branch_up:add(nn.Transpose({1,3},{1,2}))

local concat = nn.ConcatTable()
concat:add(branch_up):add(memory_branch)
model:add(concat)

-- add sampler
model:add(nn.BilinearSamplerBHWD())
model:add(nn.Transpose({1,3},{2,3})) -- untranspose the result!!

-- add spatial decoder
model:add(decoder)

-- loss module: penalise difference of gradients
local gx = torch.Tensor(3,3):zero()
gx[2][1] = -1
gx[2][2] =  0
gx[2][3] =  1
gx = gx:cuda()
local gradx = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
gradx.weight:copy(gx)
gradx.bias:fill(0)

local gy = torch.Tensor(3,3):zero()
gy[1][2] = -1
gy[2][2] =  0
gy[3][2] =  1
gy = gy:cuda()
local grady = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
grady.weight:copy(gy)
grady.bias:fill(0)

local gradconcat = nn.ConcatTable()
gradconcat:add(gradx):add(grady)

gradloss = nn.Sequential()
gradloss:add(gradconcat)
gradloss:add(nn.JoinTable(1))

criterion = nn.MSECriterion()
--criterion.sizeAverage = false

-- move everything to gpu
model:cuda()
gradloss:cuda()
criterion:cuda()
  
