--[[
  Demo script to train a model using the convolutional LSTM module 
  to predict the next frame in a sequence.
--]]

unpack = unpack or table.unpack

require 'nn'
require 'cunn'
require 'paths'
require 'torch'
require 'cutorch'
require 'image'
require 'optim'
require 'ConvLSTM'

local function main()
  cutorch.setDevice(1)
  paths.dofile('opts-mnist.lua')
  opt.untied = true
  paths.dofile('data-mnist.lua')
  paths.dofile('model-demo-ConvLSTM.lua')

  config = {}
  opt.train = true
  -----------------------------------------------------------------------------
  -- Create model or load a pre-trained one
  if opt.modelFile then -- resume training 
    model  = torch.load(opt.modelFile)
    if opt.train then
      config = torch.load(opt.configFile)
    end
  end
  
  if opt.train then
    -----------------------------------------------------------------------------
    -- Load data for training and verify one sample
    dataset = getdataSeq_mnist(opt.dataFile) -- we sample nSeq consecutive frames
    local trainSamples = dataset:size()
    print  ('main: Loaded ' .. trainSamples .. ' train sequences')
    local seq = dataset[1][1]
    print ('main: Verify sample')
    print ('main: Image size')
    print (seq[1]:size())
    print ('main: Min '.. seq[1]:min() .. ', Max ' .. seq[1]:max())

    if opt.display then
      _check_ = image.display{image=seq, win=_check_, legend='Check sample sequence', nrow = seq:size(1)}
    end
    parameters, grads = model:getParameters()
    print('Number of parameters ' .. parameters:nElement())
    print('Number of grads ' .. grads:nElement())

    local eta = config.eta or opt.eta 
    local momentum = config.momentum or opt.momentum
    local iter  = config.iter or 1 
    local epoch = config.epoch or 0
    local err  = 0
  
    model:training()
    model:forget()
    rmspropconf = {learningRate = eta}

    for t = 1,opt.maxIter do
      iter = iter+1

      --------------------------------------------------------------------
      -- define eval closure
      local feval = function()
        local f = 0
 
        model:zeroGradParameters()
        inputTable = {}
        targetTable = {}
        sample = dataset[t]
        data = sample[1]
        for i = 1,data:size(1)-1 do
          table.insert(inputTable, data[i]:cuda())
        end
        for i = 2,data:size(1) do
          table.insert(targetTable, data[i]:cuda())
        end
      
        -- estimate f and gradients
        output = model:updateOutput(inputTable)

        f = criterion:updateOutput(output,targetTable)

        -- gradients
        local gradOutput = criterion:updateGradInput(output,targetTable)
                 
        model:updateGradInput(inputTable,gradOutput)

        model:accGradParameters(inputTable, gradOutput)  

        grads:clamp(-opt.gradClip,opt.gradClip)
        return f, grads
      end
   
   
      if math.fmod(t,trainSamples) == 0 then
        epoch = epoch + 1
        eta = opt.eta*math.pow(0.5,epoch/50) 
        rmspropconf.learningRate = eta   
      end  

      _,fs = optim.rmsprop(feval, parameters, rmspropconf)

      err = err + fs[1]
      model:forget()
      --------------------------------------------------------------------
      -- compute statistics / report error
      if math.fmod(t , opt.statInterval) == 0 then
        print('==> iteration = ' .. t .. ', average loss = ' .. err/(opt.nSeq) .. ' lr '..eta ) -- err/opt.statInterval)
        err = 0
      end
      if opt.save and math.fmod(t , opt.saveInterval) == 0 then
        model.cleanState()      
        torch.save(opt.dir .. '/model_' .. t .. '.bin', model)
        config = {eta = eta, epsilon = epsilon, alpha = alpha, iter = iter, epoch = epoch}
        torch.save(opt.dir .. '/config_' .. t .. '.bin', config)
      end
      
      if opt.display and math.fmod(t , opt.displayInterval) == 0 then
        _imInput_ = image.display{image=inputTable,win = _imInput_, legend = 'Input Sequence', nrow = #inputTable}
        
        _imTarget_ = image.display{image=targetTable,win = _imTarget_, legend = 'Target Frames', nrow = #targetTable}
        _imOutput_ = image.display{image=output,win = _imOutput_, legend = 'Output', nrow = #output}
      end  
    end
  end
  print ('Training done')
  collectgarbage()

  -------------------------------------------------------------------------
  -- Evaluation mode 
  print ('Start quantitative evaluation')
  model:evaluate() 
  model:forget()
  dataset = {}
  dataset = getdataSeq_mnist(opt.dataFileTest)
  local testSamples = dataset:size()
  print  ('main: Loaded ' .. testSamples .. ' test sequences')

  err = 0
  for t = 1,testSamples do
    local f = 0
    inputTable = {}
    target  = torch.Tensor()
    sample = dataset[t]
    data = sample[1]
    for i = 1,data:size(1)-1 do
      table.insert(inputTable, data[i]:cuda())
    end
    target:resizeAs(data[1]):copy(data[data:size(1)])
    
    target = target:cuda()
    output = model:updateOutput(inputTable)
    
    f = criterion:updateOutput(output,target)

    print ('Error for image '.. t .. ' '.. f)
    if opt.display then
      _imInput_ = image.display{image=inputTable,win = _imInput_, legend = 'Input Sequence'}
      _imTarget_ = image.display{image=target:squeeze(),win = _imTarget_, legend = 'Target Frame'}
      _imOutput_ = image.display{image=output:squeeze(),win = _imOutput_, legend = 'Output'}
    end 
    err = err + f
  end
  
  print ('Average error '.. err/t)
  print ('Quantitative testing done')
  collectgarbage()
end
main()
