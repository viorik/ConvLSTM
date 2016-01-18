# ConvLSTM

Source code associated with [Spatio-temporal video autoencoder with differentiable memory](http://arxiv.org/abs/1511.06309), submitted to ICLR2016. 

This is a demo version to be trained on our modified version of moving MNIST dataset, available [here](http://mi.eng.cam.ac.uk/~vp344/). Some videos obtained on real test sequences are also available [here](http://mi.eng.cam.ac.uk/~vp344/). 

This code extends the [rnn](https://github.com/Element-Research/rnn) package by providing a spatio-temporal convolutional version of LSTM cells.

To run this demo, you first need to install the [extracunn](https://github.com/viorik/extracunn) package, which contains cuda code for SpatialConvolutionalNoBias layer and Huber gradient computation.

You also need to install the [stn](https://github.com/qassemoquab/stnbhwd) package, and replace the existing BilinearSamplerBHWD.lua with the file provided here.

More details soon.


 

