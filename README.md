# ConvLSTM

Source code associated with [Spatio-temporal video autoencoder with differentiable memory](http://arxiv.org/abs/1511.06309), submitted to ICLR2016. 

This is a demo version to be trained on our modified version of moving MNIST dataset, available [here](http://mi.eng.cam.ac.uk/~vp344/). Some videos obtained on real test sequences are also available [here](http://mi.eng.cam.ac.uk/~vp344/). 

#### Dependencies

* [rnn](https://github.com/Element-Research/rnn): our code extends [rnn](https://github.com/Element-Research/rnn) by providing a spatio-temporal convolutional version of LSTM cells.
* [extracunn](https://github.com/viorik/extracunn): contains cuda code for SpatialConvolutionalNoBias layer and Huber gradient computation.
* [stn](https://github.com/qassemoquab/stnbhwd).


 

