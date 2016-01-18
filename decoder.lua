require 'nn'

-- Decoder, mirror of the encoder, but no non-linearity
-- first layer 
decoder = nn.Sequential()
decoder:add(nn.SpatialUpSamplingNearest(2)) 
local conv = nn.Sequential()
conv:add(nn.SpatialConvolution(opt.nFilters[2], opt.nFilters[1], opt.kernelSize, opt.kernelSize, 1, 1, opt.padding, opt.padding))
local conv_new = require('weight-init')(conv, 'xavier')
decoder:add(conv_new)

