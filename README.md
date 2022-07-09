# CDL: Deep Learning in C

A hobby project I started in order to learn about neural networks.

This is an implementation of a dense neural network for classifying images of handprinted digits available in the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database), downloaded from [here](https://deepai.org/dataset/mnist).

# Why?

Just for fun. I realize that C is a bit tricky to get fun out of but I find a way.

I also wanted to practice my C and get down to the math behind DL. I wanted to build my own implementation before getting into ML using Python, for example, to do it for me.

This also opens up the possibility to use this model in a olc::PixelGameEngine project, a minimal C++ game engine I have used for several projects before, to test it against live inputs.

# Usage

For a demo just run the program without arguments to train a model with 2 hidden layers of 16 neurons for 10 epochs with 5-fold data augmentation, from the project root:\
`./build/release/cdl`\
or for debug builds:\
`./build/debug/cdl`

The full options and help can be accessed using `./cdl -h`

To compile see instructions in the src directory for the version needed, C or CUDA C.

# GPU Acceleration

For GPU acceleration I use CUDA C, a C-like language for general purpose gpu programming for NVidia cards.

Some differences will be present in some aspects of the program, like for example augmented images changing less drastically in the CUDA version, but there is a significant speed-up to consider, like for example **18x** for image augmentation.

# Samples

The program dumps a serialized version of the network into a file, two such networks are available in the samples directory, with their name suggesting how many layers (including input and output) there are and how big each one is.

A function in `nn_tools.c` is provided for loading this model into a program: `dl_load()`.