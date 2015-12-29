opt = {}
-- general options:
opt.dir     = 'outputs_mnist_line' -- subdirectory to save experiments in
opt.seed    = 1250         -- initial random seed

-- Model parameters:
opt.inputSizeW = 64   -- width of each input patch or image
opt.inputSizeH = 64   -- width of each input patch or image
opt.eta       = 1e-4 -- learning rate
opt.etaDecay  = 1e-5 -- learning rate decay
opt.momentum  = 0.9  -- gradient momentum
opt.maxIter   = 1000000 --max number of updates
opt.nSeq      = 19
opt.transf    = 2       -- number of parameters for transformation; 6 for affine or 3 for 2D transformation
opt.nFilters  = {1,32}--9,45} -- number of filters in the encoding/decoding layers
opt.nFiltersMemory   = {32,45} --{45,60}
opt.kernelSize       = 7 -- size of kernels in encoder/decoder layers
opt.kernelSizeMemory = 7
opt.kernelSizeFlow   = 15
opt.padding   = torch.floor(opt.kernelSize/2) -- pad input before convolutions
opt.dmin = -0.5
opt.dmax = 0.5
opt.gradClip = 50
opt.stride = 1 --opt.kernelSizeMemory -- no overlap
opt.constrWeight = {0,1,0.001}

opt.memorySizeW = 32
opt.memorySizeH = 32

opt.dataFile = 'dataset_fly_64x64_lines_train.t7'
opt.statInterval = 50 -- interval for printing error
opt.v            = false  -- be verbose
opt.display      = true -- display stuff
opt.displayInterval = opt.statInterval*10
opt.save         = true -- save models


if not paths.dirp(opt.dir) then
   os.execute('mkdir -p ' .. opt.dir)
end
