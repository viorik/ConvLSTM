--[[
  Convolutional LSTM for short term visual cell
  inputSize - number of input feature planes
  outputSize - number of output feature planes
  rho - recurrent sequence length
  kc  - convolutional filter size to convolve input
  km  - convolutional filter size to convolve cell; usually km > kc
  First step is untied.  
--]]

require 'ConvLSTM'

local backend_name = 'cudnn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end

local UntiedConvLSTM, parent = torch.class('nn.UntiedConvLSTM', 'nn.ConvLSTM')

function UntiedConvLSTM:__init(inputSize, outputSize, rho, kc, km, stride, batchSize)
   parent.__init(self, inputSize, outputSize, rho, kc, km, stride, batchSize)
   self.untiedModule = self:buildModelUntied()
end

function UntiedConvLSTM:buildGateUntied()
   -- Note : Input is : input(t)
   local gate = nn.Sequential()
   gate:add(backend.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc))
   gate:add(backend.Sigmoid())
   return gate
end

function UntiedConvLSTM:buildCellGateUntied()
   local cellGate = nn.Sequential()
   cellGate:add(backend.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc))
   cellGate:add(backend.Tanh())
   self.cellGateUntied = cellGate
   return cellGate
end

function UntiedConvLSTM:buildModelUntied()
   -- Input is : input(t)
   local model = nn.Sequential()
   self.inputGateUntied = self:buildGateUntied() 
   self.cellGateUntied = self:buildCellGateUntied()
   self.outputGateUntied = self:buildGateUntied()
   local concat = nn.ConcatTable()
   concat:add(self.inputGateUntied):add(self.cellGateUntied):add(self.outputGateUntied)
   model:add(concat)
   local cellAct = nn.Sequential()
   cellAct:add(nn.NarrowTable(1,2))
   cellAct:add(nn.CMulTable())
   local concat2 = nn.ConcatTable()
   concat2:add(cellAct):add(nn.SelectTable(3))
   model:add(concat2)
   local tanhcell = nn.Sequential()
   tanhcell:add(nn.SelectTable(1)):add(backend.Tanh())
   local concat3 = nn.ConcatTable()
   concat3:add(nn.SelectTable(2)):add(tanhcell):add(nn.SelectTable(1))
   model:add(concat3)
   model:add(nn.FlattenTable())
   local output = nn.Sequential()
   output:add(nn.NarrowTable(1,2))
   output:add(nn.CMulTable())
   local concat4 = nn.ConcatTable()
   concat4:add(output):add(nn.SelectTable(3))
   model:add(concat4)
   return model
end

function UntiedConvLSTM:updateOutput(input)
   local prevOutput, prevCell

   -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
   local output, cell   

   if self.step == 1 then
      if self.batchSize then
         self.zeroTensor:resize(self.batchSize,self.outputSize,input:size(3),input:size(4)):zero()
      else
         self.zeroTensor:resize(self.outputSize,input:size(2),input:size(3)):zero()
      end
      output, cell = unpack(self.untiedModule:updateOutput(input))
   else
      -- previous output and memory of this module
      prevOutput = self.outputs[self.step-1]
      prevCell = self.cells[self.step-1]
      if self.train ~= false then
         self:recycle()
         local recurrentModule = self:getStepModule(self.step)
         -- the actual forward propagation
         output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
      else
         output, cell = unpack(self.recurrentModule:updateOutput{input, prevOutput, prevCell})
      end   
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

function UntiedConvLSTM:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)
   
   -- set the output/gradOutput states of current Module
   if self.gradPrevOutput then
      self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
      nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
      gradOutput = self._gradOutputs[step]
   end

   local gradInput
   local gradInputTable
   local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
   if step == 1 then 
      gradInput = self.untiedModule:updateGradInput(input, {gradOutput, gradCell})
   else
      local recurrentModule = self:getStepModule(step)
      local output = self.outputs[step-1]
      local cell   = self.cells[step-1]
      local inputTable = {input, output, cell} 
   
      -- backward propagate through this step
      gradInputTable = recurrentModule:updateGradInput(inputTable, {gradOutput, gradCell})
      gradInput, self.gradPrevOutput, gradCell = unpack(gradInputTable)
   end
   
   self.gradCells[step-1] = gradCell
   if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end
   if self.userPrevCell then self.userGradPrevCell = gradCell end
   
   return gradInput
end

function UntiedConvLSTM:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)
   
   -- set the output/gradOutput states of current Module
   gradOutput = (step == self.step-1) and gradOutput or self._gradOutputs[step]
   gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
   gradOutputTable = {gradOutput, gradCell}

   if step == 1 then
      self.untiedModule:accGradParameters(input, gradOutputTable,scale)
   else
      local recurrentModule = self:getStepModule(step)
      local output = self.outputs[step-1]
      local cell   = self.cells[step-1]
      local inputTable = {input, output, cell}
      recurrentModule:accGradParameters(inputTable, gradOutputTable,scale)
   end   
end

function UntiedConvLSTM:initBias(forgetBias, otherBias)
  local fBias = forgetBias or 1
  local oBias = otherBias or 0
  self.inputGate.modules[2].modules[1].bias:fill(oBias)
  self.outputGate.modules[2].modules[1].bias:fill(oBias)
  self.cellGate.modules[2].modules[1].bias:fill(oBias)
  self.forgetGate.modules[2].modules[1].bias:fill(fBias)
  self.inputGateUntied.modules[1].bias:fill(oBias)
  self.outputGateUntied.modules[1].bias:fill(oBias)
  self.cellGateUntied.modules[1].bias:fill(oBias)
end
