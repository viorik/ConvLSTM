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
