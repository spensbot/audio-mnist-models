# AUDIO_MNIST Classification

A variety of simple Deep Learning models for classifying AUDIO_MNIST examples.

## audio_mnist.ipynb

Run these Jupyter notebook cells to train the models

## tools

A variety of generic tools to keep the Jupyter notebook simple and speed high-level iteration

## Pre-Processing

The torchaudio library is used to load audio data, and convert it to a low-resolution mel-spectrogram.

After pre-processing, the audio data input to the model has a very similar form to the 2D image data of the standard MNIST dataset.

## Results

Three basic model architectures were trained over the course of 50 Epochs

### Basic Feed-Forward Net

~50K Params

Train 99.05% | Test 95.80%

### 2D Convolutional Net

~25K Params

Train 97.65% | Test 94.00%

### GRU Net

~6,300 Params

Train 100.00% | Test 97.80%
