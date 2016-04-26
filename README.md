# ConvLSTM

Source code associated with [Spatio-temporal video autoencoder with differentiable memory](http://arxiv.org/abs/1511.06309), to appear in ICLR2016 Workshop track. 

This is a demo version to be trained on a modified version of moving MNIST dataset, available [here](http://mi.eng.cam.ac.uk/~vp344/). Some videos obtained on real test sequences are also available [here](http://mi.eng.cam.ac.uk/~vp344/) (not up-to-date though). 

The repository contains also a demo, main-demo-ConvLSTM.lua, of training a simple model, model-demo-ConvLSTM.lua, using the ConvLSTM module to predict the next frame in a sequence. The difference between this model and the one in the paper is that the former does not explicitly estimate the optical flow to generate the next frame. 

The ConvLSTM module can be used as is. Optionally, the untied version implemented in UntiedConvLSTM class, can be employed. The latter uses a separate model for the first step in the sequence, which has no memory. This can be helpful in training on shorter sequences, to reduce the impact of the first (memoryless) step on the training.  
 
#### Dependencies
* [rnn](https://github.com/Element-Research/rnn): our code extends [rnn](https://github.com/Element-Research/rnn) by providing a spatio-temporal convolutional version of LSTM cells.
* [extracunn](https://github.com/viorik/extracunn): contains cuda code for SpatialConvolutionalNoBias layer and Huber gradient computation.
* [stn](https://github.com/qassemoquab/stnbhwd).


 

