#pragma once
#ifndef NN_TOOLS_H
#define NN_TOOLS_H

#include <inttypes.h>

#define DL_INPUT        0
#define DL_HIDDEN       1
#define DL_OUTPUT       2

#define DL_RANDMAX      1
#define DL_RANDMIN     -1

// Data transformations
#define DL_ROTATIONMAX  0.2618 // 15 deg in radians
#define DL_SHIFTMAX     4
#define DL_SHEARMAX     0.2
// #define DL_ZOOMMIN      0.75
// #define DL_ZOOMMAX      1.25

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))
// Matrix code

// Note: the matrix itself is just pointer to pointers,
// copying the matrix or passing by reference is the same in effect
typedef struct
{
    int height;     // number of rows
    int width;      // number of cols
    double **matrix;   // malloced matrix
} matrix;

// Returns 0 matrix struct with width number of columns and height number of rows
matrix matrix_create(int height, int width);

// Frees matrix
int matrix_free(matrix A);

// Prints a matrix to console
void matrix_print(matrix A, const char *end);

// Returns matrix struct resulting of matrix multiplication
// Rows in A and columns in B have to match
matrix matrix_mult(matrix A, matrix B);

// Multiplies all elements of A by n, returns the same matrix
matrix matrix_scalar_mult(matrix A, double n);

// Adds B to A, modifies and returns A
matrix matrix_add(matrix A, matrix B);

// Substracts B from A, modifies and returns A
matrix matrix_sub(matrix A, matrix B);

// Applies sigmoid function to all values of a matrix, returns the same matrix
// Specifically, this is the tanh function modified to output within (0, 1)
matrix matrix_sigmoid(matrix A);

// Applies the derivative of the sigmoid function used in matrix_sigmoid to all
// values of a matrix, returns the same matrix
matrix matrix_derivated_sigmoid(matrix A);

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
    matrix der_last_activations, last_activations;
    // cumulative adjustments to the weights and biases
    matrix ca_weights, ca_biases;
    // Amount of adjustments
    unsigned int n_ca;
    // NULL if created, ptr if allocated
    struct node* self;
} node;

// Creates and returns a pointer to a node
// If prev is NULL, the node will be an input node, otherwise, the node pointed to by prev will point to this node as .next
// type is either DL_INPUT, DL_HIDDEN or DL_OUTPUT
node *dl_create_node(int type, int size, node *prev);

// Creates n_layers nodes + 1 input node, and returns a pointer to this input node
// n_layers does no include the input layer, but includes the output layer
// sizes is an array of layer sizes, with n_layers amount of items and the last item being the output size
node *dl_create(int n_inputs, int n_layers, int *sizes);

// Checks if the network is valid
//  0 -> not valid
//  number of layers -> valid
int dl_check(node *in_node);

// Prints the net's structure to the console
void dl_print_structure(node *head);

// Prints structure into buf up to len - 1
// Returns number of characters written
int dl_structure_str(node *head, char *buf, int len);

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
// If the magic number "CDLD" is not present, the return is NULL
node *dl_load(const char *filename);

// Create a copy of a Deep Neural Network
node *dl_copy(node *head);

// Using an input column and the neural networks input node, calculate the result column
matrix dl_process(node *in_node, matrix input);

// Calculates the cost for the result
// Cost = Sum((result - expected)^2)
double dl_cost(matrix result, matrix expected);

// Applies the stored adjustments to the weights and biases
void dl_adjust(node *head);

// Compute loss matrix, 2 * (a(i) - y(i))
matrix dl_mse_loss(matrix output, matrix expected);

// Starts the backwards pass and calculates adjustments to weights and biases for all layers but the input
// loss is the difference between output and expected
// alpha is the learning constant
void dl_backpropagation(node *head, matrix loss, double alpha);

// Data transformation code

// Randomly rotate the image to create a new one
void dl_rotate_image_rand(uint8_t *dst, uint8_t *image);

// Randomly shift the image vertically and horizontally to create a new one
void dl_shift_image_rand(uint8_t *dst, uint8_t *image);

// Randomly scale the image up or down to create a new one
// uint8_t *dl_zoom_image_rand(uint8_t *image);

// Randomly apply shear to the image to create a new one
void dl_shear_image_rand(uint8_t *dst, uint8_t *image);

#ifdef __NVCC__

#define BLOCK_SIZE 32

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error: %s at %s:%d\n",cudaGetErrorString(x),__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

// Device matrix functions
__device__ matrix d_matrix_create(int height, int width);
__device__ int d_matrix_free(matrix A);
__device__ matrix d_matrix_zero(matrix A);
__device__ matrix d_matrix_init_rand(matrix A, double min, double max, unsigned long long seed);

// Network creation for GPU
node *dl_create_node_GPU(int type, int size, node *d_prev, unsigned long long seed);
node *dl_create_GPU(int n_inputs, int n_layers, int *sizes, unsigned long long seed);
int dl_free_GPU(node *d_head);
void dl_copy_to_GPU(node *d_head, node *head);

// Network procedures for GPU
matrix dl_process_GPU(node *d_in_node, matrix input, int n_outputs);
void dl_backpropagation_GPU(node *d_head, matrix loss, double alpha);
void dl_adjust_GPU(node *d_head);

// Data augmentation
__global__ void augment_images_CUDA(uint8_t *d_images_dst, uint8_t *d_images, int img_count, int aug_factor, unsigned long long seed);

#else

#endif

#endif // NN_TOOLS_H