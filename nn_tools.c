#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "nn_tools.h"

// Returns 0 matrix struct with width number of columns and height number of rows
matrix matrix_create(int height, int width)
{
    matrix A = (matrix){.width = width, .height = height, .matrix = NULL};

    A.matrix = (double **)malloc(sizeof(double *) * A.height);
    if (!A.matrix)
    {
        fprintf(stderr, "matric_create: Allocation error\n");
        return NULL_MATRIX;
    }

    for (int i = 0; i < A.height; i++)
    {
        A.matrix[i] = (double *)malloc(sizeof(double) * A.width);
        if (!A.matrix[i])
        {
            // go back and free malloced rows
            for (i--; i >= 0; i--)
            {
                free(A.matrix[i]);
            }
            fprintf(stderr, "matric_create: Allocation error\n");
            free(A.matrix);
            return NULL_MATRIX;
        }

        for (int j = 0; j < A.width; j++)
        {
            A.matrix[i][j] = 0;
        }
    }

    return A;
}

// Frees matrix
int matrix_free(matrix A)
{
    if (A.matrix == NULL)
    {
        fprintf(stderr, "matrix_free: Matrix is NULL\n");
        return 1;
    }

    for (int i = 0; i < A.height; i++)
    {
        free(A.matrix[i]);
    }
    free(A.matrix);
    return 0;
}

// Prints a matrix to console
void matrix_print(matrix A, const char *end)
{
    for (int i = 0; i < A.height; i++)
    {
        for (int j = 0; j < A.width; j++)
        {
            printf("%lf\t", A.matrix[i][j]);
        }
        printf("\n");
    }
    if (end == NULL)
        printf("\n");
    else printf(end);
    fflush(stdout);
}

// Returns matrix struct resulting of matrix multiplication
// Columns in A and rows in B have to match
matrix matrix_mult(matrix A, matrix B)
{
    if (A.width != B.height)
    {
        fprintf(stderr, "matrix_mult: Row and column number incompatible\n");
        return NULL_MATRIX;
    }

    matrix C = matrix_create(A.height, B.width);
    int n;

    for (int r_a = 0; r_a < A.height; r_a++)
    {
        for (int c_b = 0; c_b < B.width; c_b++)
        {
            C.matrix[r_a][c_b] = 0;
            for (int c_a = 0; c_a < A.width; c_a++)
            {
                C.matrix[r_a][c_b] += A.matrix[r_a][c_a] * B.matrix[c_a][c_b];
            }
        }
    }

    return C;
}

// Adds B to A, modifies and returns A
matrix matrix_add(matrix A, matrix B)
{
    if (A.height != B.height || A.width != B.width)
    {
        fprintf(stderr, "matrix_add: Matrix dimension incompatible\n");
        return NULL_MATRIX;
    }

    for (int i = 0; i < A.height; i++)
        for (int j = 0; j < A.width; j++)
            A.matrix[i][j] = A.matrix[i][j] + B.matrix[i][j];

    return A;
}

// Applies sigmoid function to all values of a matrix, returns the same matrix
// Specifically, this is the tanh function modified to output within (0, 1)
matrix matrix_sigmoid(matrix A)
{
    for (int i = 0; i < A.height; i++)
        for (int j = 0; j < A.width; j++)
            A.matrix[i][j] = (1 + tanh(A.matrix[i][j])) / 2;

    return A;
}

/* generate a random floating point number from min to max */
double randfrom(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

// Fills a matrix with random data, returns the same matrix
matrix matrix_init_rand(matrix A, double min, double max)
{
    for (int i = 0; i < A.height; i++)
        for (int j = 0; j < A.width; j++)
            A.matrix[i][j] = randfrom(min, max);

    return A;
}

// Copies mat into A.matrix, returns the same matrix
matrix matrix_init(matrix A, double **mat)
{
    for (int i = 0; i < A.height; i++)
    {
        for (int j = 0; j < A.width; j++)
        {
            A.matrix[i][j] = mat[i][j];
        }
    }

    return A;
}

// Takes in matrix struct and returns copy
matrix matrix_copy(matrix from)
{
    matrix ret = matrix_create(from.height, from.width);
    matrix_init(ret, from.matrix);
    return ret;
}

// Sets the entire matrix to 0, returns the same matrix
matrix matrix_zero(matrix A)
{
    for (int i = 0; i < A.height; i++)
    {
        for (int j = 0; j < A.width; j++)
        {
            A.matrix[i][j] = 0;
        }
    }

    return A;
}

// Creates and returns a pointer to a node.
// If prev is NULL, the node will be an input node, otherwise, the node pointed to by prev will point to this node as .next
node *dl_create_node(int type, int size, node *prev)
{

}

// Using an input column and the neural networks input node, calculate the result column
matrix dl_process(node *in_node, matrix input)
{
    matrix result, interm;

    // Error checking
    if (in_node->prev == NULL && !dl_check(in_node))
    {
        fprintf(stderr, "dl_process: Check failed\n");
        return NULL_MATRIX;
    }
    if (in_node->prev == NULL && (input.width != 1 || input.height != in_node->weights.height))
    {
        fprintf(stderr, "dl_process: Input incompatible\n");
        return NULL_MATRIX;
    }
    
    // If the current layer is not the input layer, do calculation, else pass on the input
    if (in_node->prev != NULL)
        interm = matrix_sigmoid(matrix_add(matrix_mult(in_node->weights, input), in_node->biases));
    else
        interm = matrix_copy(input);

    // interm is the vector of neuron activations for this layer, for the head it is the input, for any other layer, the result of the calculation
    matrix_init(in_node->last_activations, interm.matrix);

    if (in_node->next != NULL)
    {
        // Recursion, go to next layer
        result = dl_process(in_node->next, interm);
        matrix_free(interm);
    } else
    {
        // Base case: output layer
        return interm;
    }

    return result;
}

// Checks if the network is valid
//  0 -> not valid
//  1 -> valid
int dl_check(node *in_node)
{
    node *this, *next;

    this = in_node;
    next = in_node->next;

    while (next != NULL)
    {
        if (next->weights.width != this->weights.height
            || this->biases.height != this->weights.height
            || this->biases.width != 1
            || this->biases.height != this->last_activations.height)
        {
            return 0;
        } else
        {
            this = next;
            next = this->next;
        }
    }

    return 1;
}

// Assembles all the nodes into a linked structure usable by dl_process and dl_check using the head node
// Calls dl_check to check for the result
//  0 -> not valid
//  1 -> valid
int dl_assemble(node *head, node **hidden_layers, int hidden_count, node *tail)
{
    head->prev = NULL;
    tail->next = NULL;
    if (hidden_count == 0)
    {
        head->next = tail;
        tail->prev = head;
    } else for (int i = 0; i < hidden_count; i++)
    {
        // assign previous layer ->next and current layer ->prev
        if (i == 0)
        {
            head->next = hidden_layers[i];
            hidden_layers[i]->prev = head;
        } else
        {
            hidden_layers[i - 1]->next = hidden_layers[i];
            hidden_layers[i]->prev = hidden_layers[i - 1];
        }

        // assign tail ->prev and current layer ->next on last iteration
        if (i == hidden_count - 1)
        {
            hidden_layers[i]->next = tail;
            tail->prev = hidden_layers[i];
        }
    }

    return dl_check(head);
}

// Frees node and linked nodes
int dl_free(node *head)
{
    matrix_free(head->biases);
    matrix_free(head->ca_biases);
    matrix_free(head->weights);
    matrix_free(head->ca_weights);
    matrix_free(head->last_activations);

    if (head->next) // if not tail
    {
        // free subsequent layers
        dl_free(head->next);
    }

    // if allocated
    if (head->self) free(head->self);
    return 0;
}

// Serializes the network and stores it in a binary file
void dl_dump(node *head, const char *filename)
{
    if (!head)
        return;

    unsigned int node_count = 1;
    FILE *fd = fopen(filename, "wb");

    // count nodes
    node *aux = head;
    while ((aux = aux->next)) node_count++;
    fwrite(&node_count, sizeof(unsigned int), 1, fd);

    aux = head;
    for (int i = 0; i < node_count; i++, aux = aux->next)
    {
        // write weights matrix
        fwrite(&aux->weights.height, sizeof(int), 1, fd);
        fwrite(&aux->weights.width, sizeof(int), 1, fd);
        for (int j = 0; j < aux->weights.height; j++)
            fwrite(aux->weights.matrix[j], sizeof(double), aux->weights.width, fd);

        // write biases matrix
        fwrite(&aux->biases.height, sizeof(int), 1, fd);
        fwrite(&aux->biases.width, sizeof(int), 1, fd);
        for (int j = 0; j < aux->biases.height; j++)
            fwrite(aux->biases.matrix[j], sizeof(double), aux->biases.width, fd);
    }

    fclose(fd);
}

// Loads a network from a file
node dl_load(const char *filename)
{
    unsigned int node_count;
    FILE *fd = fopen(filename, "rb");

    fread(&node_count, sizeof(unsigned int), 1, fd);
    
    node *nodes = (node*) malloc(sizeof(node) * node_count);
    int h, w;
    for (int i = 0; i < node_count; i++)
    {
        // Since only one malloc is used, it only needs to be freed once for the input node
        nodes[i].self = NULL;

        // write weights matrix
        fread(&h, sizeof(int), 1, fd);
        fread(&w, sizeof(int), 1, fd);
        
        nodes[i].weights = matrix_create(h, w);
        nodes[i].ca_weights = matrix_create(h, w);
        for (int r = 0; r < h; r++)
            fread(nodes[i].weights.matrix[r], sizeof(double), w, fd);

        // write biases matrix
        fread(&h, sizeof(int), 1, fd);
        fread(&w, sizeof(int), 1, fd);
        
        nodes[i].biases = matrix_create(h, w);
        nodes[i].ca_biases = matrix_create(h, w);
        nodes[i].last_activations = matrix_create(h, w);
        for (int r = 0; r < h; r++)
            fread(nodes[i].biases.matrix[r], sizeof(double), w, fd);
    }

    node **hidden = (node**) malloc(sizeof(node*) * (node_count - 2));
    for (int i = 0; i < node_count - 2; i++)
    {
        hidden[i] = &nodes[i + 1];
    }
    dl_assemble(&nodes[0], hidden, node_count - 2, &nodes[node_count - 1]);
    free(hidden);

    fclose(fd);
    node ret = nodes[0];
    ret.self = nodes;
    return ret;
}

// Calculates the cost for the result
// Cost = Sum((result - expected)^2)
double dl_cost(matrix result, matrix expected)
{
    double cost = 0;
    double difference;

    for (int i = 0; i < result.height; i++)
    {
        difference = expected.matrix[i][0] - result.matrix[i][0];
        cost += difference * difference;
    }

    return cost;
}

// Applies the stored adjustments to the weights and biases
void dl_adjust(node *head)
{
    if (head == NULL) return;

    if (head->prev) // not the input layer
    {
        matrix_add(head->weights, head->ca_weights);
        matrix_add(head->biases, head->ca_biases);
        matrix_zero(head->ca_weights);
        matrix_zero(head->ca_biases);
    } else if (!dl_check(head))
    {
        fprintf(stderr, "dl_adjust: Adjustment matrices incompatible\n");
        return;
    }
    dl_adjust(head->next);
}

