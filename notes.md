# Neural networks
Might condense all tools into a single header file to use in other projects, like olc\
Might also make several types of neural networks, defined by the training method:
- deep learning, e.g. learning to recognize a number with a training set
- evolution, generating many iterations which compete and reproduce with mutations
- others?

## Deep learning
### DL: Basics
```
O
O   o   o
O   o   o   O
O   o   o   O
O   o   o
O

I   L1  L2  O
```

Input = vector of I input values\
Output = vector of O output values

Hidden layers represent matrix multiplications to go from input to output.

For the example above:\
Input => 6x1 column vector\
L1 => 4x6 matrix\
L2 => 4x4 matrix\
Output => 2x1 column vector\

Transitions:\
L1:\
    L1 * I + B1 => [4x6][6x1] +  [4x1] = [4x1]\
    L2 * L1 + B2 => [4x4][4x1] + [4x1] = [4x1]\
    O * L2 + B3 => [2x4][4x1] + [2x1] = [2x1]

### DL: Network representation
#### Linked-structure approach

The input layer points to the first hidden layer, which points to the next, and so on, and the last hidden layer points to the output layer.

##### File format

Magic number first: 4 bytes, equal to ASCII "CDLD"

Binary. Nodes can be read in order and assembled on loading, since pointers will need to be recalculated.

An unsigned integer determines the number of nodes to be read.\
For each node there are 2 matrices, before each one 2 integers define the dimensions, then an array of doubles can be read. A given position (r, c) can be accessed with `[c + r * width]`

Data:
- uint: node count
- node:
  - matrix: weights:
    - int: height
    - int: width
    - double[height * width]: matrix
  - matrix: biases:
    - int: height
    - int: width
    - double[height * width]: matrix

### DL: Improving performance

#### 1. Data Augmentation

Increase the number of train data in the following ways

Random rotation: each image rotates randomly

Random shift: each image randomly moves

Random shear: each image gets slightly distorted

Random zoom: each image is slightly scaled down or up