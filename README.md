# CDL: Deep Learning in C

A hobby project I started in order to learn about neural networks.

This is an implementation of a dense neural network for classifying images of handprinted digits available in the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database), downloaded from [here](https://deepai.org/dataset/mnist).

# Why?

Just for fun. I realize that C is a bit tricky to get fun out of but I find a way.

I also wanted to practice my C and get down to the math behind DL. I wanted to build my own implementation before getting into ML using Python, for example, to do it for me.

This also opens up the possibility to use this model in a olc::PixelGameEngine project, a minimal C++ game engine I have used for several projects before, to test it against live inputs.

# Usage

Initially the program is limited to hard-coded structures, specifically a 2-hidden-layer setup.

I plan to add command-line parameters to customize the model trained, and several other utilities to test existing networks.

To compile I used\
```gcc -o cdl -g test_dl.c nn_tools.c -lm -O2```\
but the flags `-g` and `-O2` are optional, for debugging and optimizations respectively.

# Samples

The program dumps a serialized version of the network into a file, two such networks are available in the samples directory, with their name suggesting how many layers (including input and output) there are and how big each one is.

A function in `nn_tools.c` is provided for loading this model into a program: `dl_load()`.