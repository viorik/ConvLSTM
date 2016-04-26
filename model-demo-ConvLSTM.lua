require 'nn'
require 'rnn'

local backend_name = 'cudnn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end

if opt.untied then 
  require 'UntiedConvLSTM'
else
  require 'ConvLSTM'
end 

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'cudnn.SpatialConvolution'
  init'nn.SpatialConvolution'
end 

net = nn.Sequential()

-- Spatial encoder
encoder = nn.Sequential()
encoder:add(backend.SpatialConvolution(opt.nFilters[1], opt.nFilters[2], opt.kernelSize, opt.kernelSize, 1, 1, opt.padding, opt.padding))
encoder:add(backend.Tanh())
encoder:add(nn.SpatialMaxPooling(2,2))
net:add(encoder)

-- Temporal encoder
if opt.untied then
  net:add(nn.UntiedConvLSTM(opt.nFiltersMemory[1],opt.nFiltersMemory[2], opt.nSeq, opt.kernelSize, opt.kernelSizeMemory, opt.stride))
else
  net:add(nn.UntiedConvLSTM(opt.nFiltersMemory[1],opt.nFiltersMemory[2], opt.nSeq, opt.kernelSize, opt.kernelSizeMemory, opt.stride))
end
  
-- Spatial decoder
decoder = nn.Sequential()
decoder:add(nn.SpatialUpSamplingNearest(2)) 
decoder:add(backend.SpatialConvolution(opt.nFiltersMemory[2], opt.nFilters[1], opt.kernelSize, opt.kernelSize, 1, 1, opt.padding, opt.padding))
net:add(decoder)

-- Init model
net:add(nn.Sigmoid())
MSRinit(net)
local lstm_paramsE, lstm_gradsE = net.modules[2]:getParameters()
lstm_paramsE:uniform(-0.08,0.08)
net.modules[2]:initBias(0,0)

-- Unroll over time using sequencer
model = nn.Sequencer(net)
model:remember('both')
model:training()

-- Loss module
criterion = nn.SequencerCriterion(nn.BCECriterion())

-- move everything to gpu
model:cuda()
criterion:cuda()
