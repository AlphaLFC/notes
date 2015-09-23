# Deep Speach


## Goals

* Simple -- deep learning end-to-end
* Fast -- multiple CPUs and network fits GPUs
* Noise-tolerant -- synthetic training data

## The Network

* Input: spectrograms of nearby frames.
* Output: alphabet distribution of the current frame.
* Layers: 3 non-recurrent layers, a recurrent layers and a non-recurrent layer.

### Efficiency

* No LSTM gates, but all rectified-linear (ReLu) gates -- few BLAS operations on GPU.
* A single recurrent layer.
* The recurrent layer divided into forward and backward parts -- forward and backward pases can be done on two GPUs simultaneiously.

### Training

* A softmax layer on top that outputs alphabet distributions.
* CTC loss L(y',y).
* Nesterov's accelerated gradient method.

### Regularization

* dropout
* jittering

### Language Model

* KenLM toolkit for language model training
* Q(c) = log P(c|x) + a log P(c|lm) + b word_count(c)
* optimized beam search algorithm

## Optimization

### Data Parallelism

* The t-th frame of many utterances is process as a batch: `W * H_t = W * [h_1,t, h_2,t, ...]`
* Aggregate gradients from multiple GPUs

### Model Parallelism

* Divide the model along the time into N parts, suppose we have N GPUs.
* At epoch n in [1,N], GPU i is processing part (i+n)%N.

### Striding

Stride of 2.

## Training Data

### Synthesis by Superpoisiton

* Many noise sources from public videos. (DNN can learn to ignore a single noise source.)
* Superposition capture audio with these many noise sources.
* Reject synthetic audio clip if the average power in each frequency differs significantly from the average pwoer observed in real noisy recordings.

### Capturing Lombard Effect

* Play noice through headphones of recorders.


