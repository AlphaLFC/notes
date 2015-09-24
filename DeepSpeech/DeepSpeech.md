# Deep Speach

This Markdown document contains LaTeX math. Please install this Chrome
browser
[plugin](https://chrome.google.com/webstore/detail/markdown-reader/gpoigdifkoadgajcincpilkjmejcaanc?hl=en)
to read it.


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

* The t-th frame of many utterances is process as a batch: $W * H_t = W * [h_1, h_2, ...]$
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


## Questions and Read More

1. ~~What is CTC loss?~~

   CTC loss is the error rate defined in Section 2. of the
   [paper](ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf) "Connectionist
   Temporal Classification" as

   $$L(h,S) = \frac{1}{|S|} \sum_{(x,z)\in S} \frac{E(h(x),z)}{|z|}$$

   where $S$ is the test set, $(x,z)$ is a pair of input and label
   sequences in $S$, and $E(.,.)$ denotes the editing distance.

1. ~~~What is Jittering?~~~

   Jittering means that during network evaluation, we shift the audio
   clip to left and right for a small time period, and feed to the
   network and average the output.

1. ~~~What is dropout?~~~

   Hinton described this method in his
   [paper](http://arxiv.org/pdf/1207.0580.pdf). This method randomly
   "dropout" some inputs to control overfitting.

1. Is the backward recurrence important?
1. With a single direction, can we pipeline recording with recognition?
1. Why this specific topology (5 layers and first 3 are non-recurrent)?
1. Why softmax in computing the probability distribution over alphabet?
1. What is Nesterov's accelerated gradient method?
1. Does the same approach works with Chinese?
1. How is the beam search algorithm works to combine DNN with the langauge model?


