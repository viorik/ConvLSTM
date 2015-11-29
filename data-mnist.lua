local data_verbose = false

function getdataSeq_mnist(datafile)
   local data = torch.DiskFile(datafile,'r'):readObject()
   local datasetSeq ={}
   data = data:float()/255.0
--   local std = std or 0.2
   local nsamples = data:size(1)
   local nseq  = data:size(2)
   local nrows = data:size(4)
   local ncols = data:size(5)
   print (nsamples .. ' ' .. nseq .. ' ' .. nrows .. ' ' .. ncols)
   function datasetSeq:size()
      return nsamples
   end

   function datasetSeq:selectSeq()
      local imageok = false
      if simdata_verbose then
         print('selectSeq')
      end
      while not imageok do
         local i = math.ceil(torch.uniform(1e-12,nsamples))
         --image index
                 
         local im = data:select(1,i)
         return im,i
      end
   end

   dsample = torch.Tensor(nseq,1,nrows,ncols)
   
   setmetatable(datasetSeq, {__index = function(self, index)
                                       local sample,i = self:selectSeq()
                                       dsample:copy(sample)
                                       return {dsample,i}
                                    end})
   return datasetSeq
end

function getdataSeq_small(datafile, nseq)
   local data = torch.DiskFile(datafile,'r'):readObject()
   --local img = torch.load(datafile)
   --local data = img.data:float() 
   --print (data:size())
   --print (type(data))
   local datasetSeq ={}

--   local std = std or 0.2
   local nsamples = data:size(1)
--   local nchannels = data:size(2)
   local nrows = data:size(3)
   local ncols = data:size(4)
--   print (nsamples .. ' ' .. nchannels .. ' ' .. nrows .. ' ' .. ncols)
   function datasetSeq:size()
      return nsamples
   end

   function datasetSeq:selectSeq(index,nseq)
      local imageok = false
      if simdata_verbose then
         print('selectSeq')
      end
      while not imageok do
         --image index
         local i = math.max(math.fmod(index,nsamples-nseq-2),1)
         --local i = math.ceil(torch.uniform(1e-12,nsamples-nseq+1))
         local im = data:narrow(1,i,nseq)
         return im,i
      end
   end

   dsample = torch.Tensor(nseq,1,nrows,ncols)
   
   setmetatable(datasetSeq, {__index = function(self, index)
                                       local sample,i = self:selectSeq(index, nseq)
                                       dsample:copy(sample)
                                       return {dsample,i}
                                    end})
   return datasetSeq
end

function displayData(dataset, nsamples, nrow, zoom)
   require 'image'
   local nsamples = nsamples or 49
   local zoom = zoom or 1
   local nrow = nrow or 4

   cntr = 1
   local ex = {}
   for i=1,nsamples do
      local exx = dataset[1]
      ex[cntr] = exx[1]:clone():reshape(96,324)
      cntr = cntr + 1
   end

   return image.display{image=ex, padding=2, legend='Training Data'}
end