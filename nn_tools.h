#ifndef NN_TOOLS_H
#define NN_TOOLS_H


#define DL_INPUT    0
#define DL_HIDDEN   1
#define DL_OUTPUT   2

#define DL_RANDMAX 1
#define DL_RANDMIN -1

// Matrix code

// Note: the matrix itself is just pointer to pointers,
// copying the matrix or passing by reference is the same in effect
typedef struct
{
    int height;     // number of rows
    int width;      // number of cols
    double **matrix;   // malloced matrix
} matrix;

#define NULL_MATRIX (matrix) { .width = 0, .height = 0, .matrix = NULL }

// Returns 0 matrix struct with width number of columns and height number of rows
matrix matrix_create(int height, int width);

// Frees matrix
int matrix_free(matrix A);

// Prints a matrix to console
void matrix_print(matrix A, const char *end);

// Returns matrix struct resulting of matrix multiplication
// Rows in A and columns in B have to match
matrix matrix_mult(matrix A, matrix B);

// Adds B to A, modifies and returns A
matrix matrix_add(matrix A, matrix B);

// Applies sigmoid function to all values of a matrix, returns the same matrix
// Specifically, this is the logistic function
matrix matrix_sigmoid(matrix A);

// Fills a matrix with random data, returns the same matrix
matrix matrix_init_rand(matrix A, double min, double max);

// Copies mat into A.matrix, returns the same matrix
matrix matrix_init(matrix A, double **mat);

// Takes in matrix struct and returns copy
matrix matrix_copy(matrix from);

// Sets the entire matrix to 0, returns the same matrix
matrix matrix_zero(matrix A);

// Network code

typedef struct node
{
    struct node* prev;
    struct node* next;
    // Neurons are an abstraction, one row per neuron
    matrix weights, biases;
    // need to keep these for backpropagation and actually training the network
    matrix last_activations;
    // cumulative adjustments to the weights and biases
    matrix ca_weights, ca_biases;
    // NULL if created, ptr if allocated
    struct node* self;
} node;

// Creates and returns a pointer to a node
// If prev is NULL, the node will be an input node, otherwise, the node pointed to by prev will point to this node as .next
// type is either DL_INPUT, DL_HIDDEN or DL_OUTPUT
node *dl_create_node(int type, int size, node *prev);

// Using an input column and the neural networks input node, calculate the result column
matrix dl_process(node *in_node, matrix input);

// Checks if the network is valid
//  0 -> not valid
//  number of layers -> valid
int dl_check(node *in_node);

// Assembles all the nodes into a linked structure usable by nn_process and nn_check using the head node
// Calls nn_check to check for the result
//  0 -> not valid
//  number of layers -> valid
int dl_assemble(node *head, node **hidden_layers, int hidden_count, node *tail);

// Frees node and subsequent linked nodes
int dl_free(node *head);

// Serializes the network and stores it in a binary file
void dl_dump(node *head, const char *filename);

// Loads a network from a file
node dl_load(const char *filename);

// Calculates the cost for the result
double dl_cost(matrix result, matrix expected);

// Applies the stored adjustments to the weights and biases
void dl_adjust(node *head);

#endif // NN_TOOLS_H