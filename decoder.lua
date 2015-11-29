require 'nn'
require 'SpatialUnPooling'

-- Decoder, mirror of the encoder, but no non-linearity
-- first layer 
decoder = nn.Sequential()
--decoder:add(nn.SpatialUpSamplingNearest(2))
--decoder:add(nn.SpatialConvolution(opt.nFilters[3], opt.nFilters[2], opt.kernelSize, opt.kernelSize, 1, 1, opt.padding, opt.padding))
--decoder:add(nn.Diag(opt.nFilters[2]))

-- second layer 

--decoder:add(nn.Dropout(0.5))
--decoder:add(nn.SpatialUnPooling(2))
decoder:add(nn.SpatialUpSamplingNearest(2)) 
local conv = nn.Sequential()
conv:add(nn.SpatialConvolution(opt.nFilters[2], opt.nFilters[1], opt.kernelSize, opt.kernelSize, 1, 1, opt.padding, opt.padding))
local conv_new = require('weight-init')(conv, 'xavier')
decoder:add(conv_new)
--decoder:add(nn.SpatialUpSamplingNearest(2)) 
--decoder:add(nn.Diag(opt.nFilters[1]))

