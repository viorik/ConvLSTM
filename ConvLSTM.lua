--[[
  Convolutional LSTM for short term visual cell
  inputSize - number of input feature planes
  outputSize - number of output feature planes
  rho - recurrent sequence length
  kc  - convolutional filter size to convolve input
  km  - convolutional filter size to convolve cell; usually km > kc  
--]]
local _ = require 'moses'
require 'nn'
require 'dpnn'
require 'rnn'
require 'extracunn'
--require 'recursiveUtils'
local ConvLSTM, parent = torch.class('nn.ConvLSTM', 'nn.AbstractRecurrent')

function ConvLSTM:__init(inputSize, outputSize, rho, kc, km, stride)
   parent.__init(self, rho or 10)
   self.inputSize = inputSize
   self.outputSize = outputSize
   self.kc = kc
   self.km = km
   self.padc = torch.floor(kc/2)
   self.padm = torch.floor(km/2)
   self.stride = stride or 1
   
   -- build the model
   self.recurrentModule = self:buildModel()
   -- make it work with nn.Container
   self.modules[1] = self.recurrentModule
   self.sharedClones[1] = self.recurrentModule 
   
   -- for output(0), cell(0) and gradCell(T)
   self.zeroTensor = torch.Tensor() 
   
   self.cells = {}
   self.gradCells = {}
end

-------------------------- factory methods -----------------------------
function ConvLSTM:buildGate()
   -- Note : Input is : {input(t), output(t-1), cell(t-1)}
   local gate = nn.Sequential()
   gate:add(nn.NarrowTable(1,2)) -- we don't need cell here
   local input2gate = nn.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)
   local output2gate = nn.SpatialConvolutionNoBias(self.outputSize, self.outputSize, self.km, self.km, self.stride, self.stride, self.padm, self.padm)
   local para = nn.ParallelTable()
   para:add(input2gate):add(output2gate) 
   gate:add(para)
   gate:add(nn.CAddTable())
   gate:add(nn.Sigmoid())
   return gate
end

function ConvLSTM:buildInputGate()
   self.inputGate = self:buildGate()
   return self.inputGate
end

function ConvLSTM:buildForgetGate()
   self.forgetGate = self:buildGate()
   return self.forgetGate
end

function ConvLSTM:buildcellGate()
   -- Input is : {input(t), output(t-1), cell(t-1)}, but we only need {input(t), output(t-1)}
   local hidden = nn.Sequential()
   hidden:add(nn.NarrowTable(1,2))
   local input2gate = nn.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)
   local output2gate = nn.SpatialConvolutionNoBias(self.outputSize, self.outputSize, self.km, self.km, self.stride, self.stride, self.padm, self.padm)
   local para = nn.ParallelTable()
   para:add(input2gate):add(output2gate)
   hidden:add(para)
   hidden:add(nn.CAddTable())
   hidden:add(nn.Tanh())
   self.cellGate = hidden
   return hidden
end

function ConvLSTM:buildcell()
   -- Input is : {input(t), output(t-1), cell(t-1)}
   self.inputGate = self:buildInputGate() 
   self.forgetGate = self:buildForgetGate()
   self.cellGate = self:buildcellGate()
   -- forget = forgetGate{input, output(t-1), cell(t-1)} * cell(t-1)
   local forget = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(self.forgetGate):add(nn.SelectTable(3))
   forget:add(concat)
   forget:add(nn.CMulTable())
   -- input = inputGate{input(t), output(t-1), cell(t-1)} * cellGate{input(t), output(t-1), cell(t-1)}
   local input = nn.Sequential()
   local concat2 = nn.ConcatTable()
   concat2:add(self.inputGate):add(self.cellGate)
   input:add(concat2)
   input:add(nn.CMulTable())
   -- cell(t) = forget + input
   local cell = nn.Sequential()
   local concat3 = nn.ConcatTable()
   concat3:add(forget):add(input)
   cell:add(concat3)
   cell:add(nn.CAddTable())
   self.cell = cell
   return cell
end   
   
function ConvLSTM:buildOutputGate()
   self.outputGate = self:buildGate()
   return self.outputGate
end

-- cell(t) = cell{input, output(t-1), cell(t-1)}
-- output(t) = outputGate{input, output(t-1)}*tanh(cell(t))
-- output of Model is table : {output(t), cell(t)} 
function ConvLSTM:buildModel()
   -- Input is : {input(t), output(t-1), cell(t-1)}
   self.cell = self:buildcell()
   self.outputGate = self:buildOutputGate()
   -- assemble
   local concat = nn.ConcatTable()
   concat:add(nn.NarrowTable(1,2)):add(self.cell)
   local model = nn.Sequential()
   model:add(concat)
   -- output of concat is {{input(t), output(t-1)}, cell(t)}, 
   -- so flatten to {input(t), output(t-1), cell(t)}
   model:add(nn.FlattenTable())
   local cellAct = nn.Sequential()
   cellAct:add(nn.SelectTable(3))
   cellAct:add(nn.Tanh())
   local concat3 = nn.ConcatTable()
   concat3:add(self.outputGate):add(cellAct)
   local output = nn.Sequential()
   output:add(concat3)
   output:add(nn.CMulTable())
   -- we want the model to output : {output(t), cell(t)}
   local concat4 = nn.ConcatTable()
   concat4:add(output):add(nn.SelectTable(3))
   model:add(concat4)
   return model
end

------------------------- forward backward -----------------------------
function ConvLSTM:updateOutput(input)
   local prevOutput, prevCell
   
   if self.step == 1 then
      prevOutput = self.userPrevOutput or self.zeroTensor
      prevCell = self.userPrevCell or self.zeroTensor
      self.zeroTensor:resize(self.outputSize,input:size(2),input:size(3)):zero()
   else
      -- previous output and memory of this module
      prevOutput = self.output
      prevCell   = self.cell
   end
      
   -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
   local output, cell
   if self.train ~= false then
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      -- the actual forward propagation
      output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
   else
      output, cell = unpack(self.recurrentModule:updateOutput{input, prevOutput, prevCell})
   end
   
   if self.train ~= false then
      local input_ = self.inputs[self.step]
      self.inputs[self.step] = self.copyInputs 
         and nn.rnn.recursiveCopy(input_, input) 
         or nn.rnn.recursiveSet(input_, input)     
   end
   
   self.outputs[self.step] = output
   self.cells[self.step] = cell
   
   self.output = output
   self.cell = cell
   
   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   self.gradParametersAccumulated = false
   -- note that we don't return the cell, just the output
   return self.output
end

function ConvLSTM:backwardThroughTime(timeStep, rho)
   assert(self.step > 1, "expecting at least one updateOutput")
   self.gradInputs = {} -- used by Sequencer, Repeater
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   
   if self.fastBackward then
      for step=timeStep-1,math.max(stop,1),-1 do
         -- set the output/gradOutput states of current Module
         local recurrentModule = self:getStepModule(step)
         
         -- backward propagate through this step
         local gradOutput = self.gradOutputs[step]
         if self.gradPrevOutput then
            self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
            nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
            gradOutput = self._gradOutputs[step]
         end
         
         local scale = self.scales[step]
         local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
         local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
         local inputTable = {self.inputs[step], output, cell}
         local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
         local gradInputTable = recurrentModule:backward(inputTable, {gradOutput, gradCell}, scale)
         gradInput, self.gradPrevOutput, gradCell = unpack(gradInputTable)
         self.gradCells[step-1] = gradCell
         table.insert(self.gradInputs, 1, gradInput)
         if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end
      end
      self.gradParametersAccumulated = true
      return gradInput
   else
      local gradInput = self:updateGradInputThroughTime()
      self:accGradParametersThroughTime()
      return gradInput
   end
end

function ConvLSTM:updateGradInputThroughTime(timeStep, rho)
   assert(self.step > 1, "expecting at least one updateOutput")
   self.gradInputs = {}
   local gradInput
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho

   for step=timeStep-1,math.max(stop,1),-1 do
      -- set the output/gradOutput states of current Module
      local recurrentModule = self:getStepModule(step)
      
      -- backward propagate through this step
      local gradOutput = self.gradOutputs[step]
      if self.gradPrevOutput then
         self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
         nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
         gradOutput = self._gradOutputs[step]
      end
      
      local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
      local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
      local inputTable = {self.inputs[step], output, cell}
      local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
      local gradInputTable = recurrentModule:updateGradInput(inputTable, {gradOutput, gradCell})
      gradInput, self.gradPrevOutput, gradCell = unpack(gradInputTable)
      self.gradCells[step-1] = gradCell
      table.insert(self.gradInputs, 1, gradInput)
      if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end
   end
   
   return gradInput
end

function ConvLSTM:accGradParametersThroughTime(timeStep, rho)
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   
   for step=timeStep-1,math.max(stop,1),-1 do
      -- set the output/gradOutput states of current Module
      local recurrentModule = self:getStepModule(step)
      
      -- backward propagate through this step
      local scale = self.scales[step]
      local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
      local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
      local inputTable = {self.inputs[step], output, cell}
      local gradOutput = (step == self.step-1) and self.gradOutputs[step] or self._gradOutputs[step]
      local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
      local gradOutputTable = {gradOutput, gradCell}
      recurrentModule:accGradParameters(inputTable, gradOutputTable, scale)
   end
   
   self.gradParametersAccumulated = true
   return gradInput
end

function ConvLSTM:accUpdateGradParametersThroughTime(lr, timeStep, rho)
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   
   for step=timeStep-1,math.max(stop,1),-1 do
      -- set the output/gradOutput states of current Module
      local recurrentModule = self:getStepModule(step)
      
      -- backward propagate through this step
      local scale = self.scales[step] 
      local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
      local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
      local inputTable = {self.inputs[step], output, cell}
      local gradOutput = (step == self.step-1) and self.gradOutputs[step] or self._gradOutputs[step]
      local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
      local gradOutputTable = {self.gradOutputs[step], gradCell}
      recurrentModule:accUpdateGradParameters(inputTable, gradOutputTable, lr*scale)
   end
   
   return gradInput
end


function ConvLSTM:initBias(forgetBias, otherBias)
  local fBias = forgetBias or 1
  local oBias = otherBias or 0
  self.inputGate.modules[2].modules[1].bias:fill(oBias)
  --self.inputGate.modules[2].modules[2].bias:fill(oBias)
  self.outputGate.modules[2].modules[1].bias:fill(oBias)
  --self.outputGate.modules[2].modules[2].bias:fill(oBias)
  self.cellGate.modules[2].modules[1].bias:fill(oBias)
  --self.cellGate.modules[2].modules[2].bias:fill(oBias)
  self.forgetGate.modules[2].modules[1].bias:fill(fBias)
  --self.forgetGate.modules[2].modules[2].bias:fill(fBias)
end
