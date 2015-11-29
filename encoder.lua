require 'nn'
require 'unsupgpu'

-- Encoder
encoder = nn.Sequential()
local conv = nn.Sequential()
conv:add(nn.SpatialConvolution(opt.nFilters[1], opt.nFilters[2], opt.kernelSize, opt.kernelSize, 1, 1, opt.padding, opt.padding))
local conv_new = require('weight-init')(conv, 'xavier')
encoder:add(conv_new)
encoder:add(nn.Tanh())
--encoder:add(nn.SmoothHuberPenalty(opt.nFilters[2], opt.constrWeight[1]))
--encoder:add(nn.L1Penalty(0.01)) 
encoder:add(nn.SpatialMaxPooling(2,2))

