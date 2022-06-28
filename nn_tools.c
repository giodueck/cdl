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
    else
        printf(end);
    fflush(stdout);
}

// Returns matrix struct resulting of matrix multiplication
// Columns in A and rows in B have to match
matrix matrix_mult(matrix A, matrix B)
{
    if (A.width != B.height)
    {
        fprintf(stderr, "matrix_mult: Column and row numbers incompatible: %d and %d.\n", A.width, B.height);
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

// Multiplies all elements of A by n, returns the same matrix
matrix matrix_scalar_mult(matrix A, double n)
{
    for (int i = 0; i < A.height; i++)
    {
        for (int j = 0; j < A.width; j++)
        {
            A.matrix[i][j] *= n;
        }
    }

    return A;
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
            A.matrix[i][j] += B.matrix[i][j];

    return A;
}

// Substracts B from A, modifies and returns A
matrix matrix_sub(matrix A, matrix B)
{
    if (A.height != B.height || A.width != B.width)
    {
        fprintf(stderr, "matrix_sub: Matrix dimension incompatible\n");
        return NULL_MATRIX;
    }

    for (int i = 0; i < A.height; i++)
        for (int j = 0; j < A.width; j++)
            A.matrix[i][j] -= B.matrix[i][j];

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

// Applies the derivative of the sigmoid function used in matrix_sigmoid to all
// values of a matrix, returns the same matrix
matrix matrix_derivated_sigmoid(matrix A)
{
    for (int i = 0; i < A.height; i++)
        for (int j = 0; j < A.width; j++)
        {
            double r = tanh(A.matrix[i][j]);
            A.matrix[i][j] = (1 - r*r) / 2;
        }

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

// Creates and returns a pointer to a node
// If prev is NULL, the node will be an input node, otherwise, the node pointed to by prev will point to this node as .next
// type is either DL_INPUT, DL_HIDDEN or DL_OUTPUT
node *dl_create_node(int type, int size, node *prev)
{
    if (type != DL_INPUT && prev == NULL)
    {
        fprintf(stderr, "dl_create_node: Previous layer not given.\n");
        exit(1);
    }

    node *n = (node *)malloc(sizeof(node));
    n->next = NULL; // will be overwritten if a node is created with n as prev
    n->self = n;
    n->n_ca = 0;

    switch (type)
    {
    case DL_INPUT:
        n->weights = matrix_create(size, 1);
        n->ca_weights = matrix_create(size, 1);
        n->biases = matrix_create(size, 1);
        n->ca_biases = matrix_create(size, 1);
        n->last_activations = matrix_create(size, 1);
        n->der_last_activations = matrix_create(size, 1);
        n->prev = NULL;
        break;

    case DL_HIDDEN:
    case DL_OUTPUT:
        n->weights = matrix_init_rand(matrix_create(size, prev->weights.height), DL_RANDMIN, DL_RANDMAX);
        n->biases = matrix_init_rand(matrix_create(size, 1), DL_RANDMIN, DL_RANDMAX);
        n->ca_weights = matrix_create(size, prev->weights.height);
        n->ca_biases = matrix_create(size, 1);
        n->last_activations = matrix_create(size, 1);
        n->der_last_activations = matrix_create(size, 1);
        n->prev = prev;
        prev->next = n->self;
        break;

    default:
        fprintf(stderr, "dl_create_node: Invalid type.\n");
        exit(1);
    }

    return n;
}

// Creates n_layers nodes + 1 input node, and returns a pointer to this input node
// n_layers does no include the input layer, but includes the output layer
// sizes is an array of layer sizes, with n_layers amount of items and the last item being the output size
node *dl_create(int n_inputs, int n_layers, int *sizes)
{
    node **nodes = (node **)malloc(sizeof(node*) * (n_layers + 1));

    nodes[0] = dl_create_node(DL_INPUT, n_inputs, NULL);

    int i;
    for (i = 0; i < n_layers - 1; i++)
    {
        nodes[i + 1] = dl_create_node(DL_HIDDEN, sizes[i], nodes[i]);
    }
    nodes[i + 1] = dl_create_node(DL_OUTPUT, sizes[i], nodes[i]);

    return nodes[0];
}

// Using an input column and the neural networks input node, calculate the result column
matrix dl_process(node *in_node, matrix input)
{
    matrix result, interm, der_interm;

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
    {
        // interm is the vector of neuron activations for this layer,
        // for the head it is the input, for any other layer, the result of the calculation
        interm = matrix_sigmoid(matrix_add(matrix_mult(in_node->weights, input), in_node->biases));

        // der_interm, derivated intermediate, is used for backpropagation and not needed for the input layer
        der_interm = matrix_derivated_sigmoid(matrix_add(matrix_mult(in_node->weights, input), in_node->biases));
        matrix_init(in_node->der_last_activations, der_interm.matrix);
        matrix_free(der_interm);
    }
    else
        interm = matrix_copy(input);

    // interm is the vector of neuron activations for this layer, for the head it is the input, for any other layer, the result of the calculation
    matrix_init(in_node->last_activations, interm.matrix);

    if (in_node->next != NULL)
    {
        // Recursion, go to next layer
        result = dl_process(in_node->next, interm);
        matrix_free(interm);
    }
    else
    {
        // Base case: output layer
        return interm;
    }

    return result;
}

// Checks if the network is valid
//  0 -> not valid
//  number of layers -> valid
int dl_check(node *in_node)
{
    node *this, *next;

    this = in_node;
    next = in_node->next;

    int i = 0;

    while (next != NULL)
    {
        if (next->weights.width != this->weights.height
        || this->biases.height != this->weights.height
        || this->biases.width != 1
        || this->biases.height != this->last_activations.height
        || this->ca_weights.height != this->weights.height
        || this->ca_weights.width != this->weights.width
        || this->ca_biases.height != this->biases.height
        || this->ca_biases.width != this->biases.width
        || this->der_last_activations.height != this->last_activations.height)
        {
            if (next->weights.width != this->weights.height)
            {
                fprintf(stderr, "dl_check: layer %d: weights matrix mismatch: %d height to %d width.\n", i, this->weights.height, next->weights.width);
            }
            if (this->biases.height != this->weights.height)
            {
                fprintf(stderr, "dl_check: layer %d: biases to weights matrix mismatch: %d height to %d height.\n", i, this->biases.height, this->weights.height);
            }
            if (this->biases.width != 1)
            {
                fprintf(stderr, "dl_check: layer %d: biases matrix too wide: %d.\n", i, this->biases.width);
            }
            if (this->biases.height != this->last_activations.height)
            {
                fprintf(stderr, "dl_check: layer %d: biases to last_activations matrix mismatch: %d height to %d height.\n", i, this->biases.height, this->last_activations.height);
            }
            if (this->ca_weights.height != this->weights.height)
            {
                fprintf(stderr, "dl_check: layer %d: ca_weights to weights matrix mismatch: %d height to %d height.\n", i, this->ca_weights.height, this->weights.height);
            }
            if (this->ca_weights.width != this->weights.width)
            {
                fprintf(stderr, "dl_check: layer %d: ca_weights to weights matrix mismatch: %d width to %d width.\n", i, this->ca_weights.width, this->weights.width);
            }
            if (this->ca_biases.height != this->biases.height)
            {
                fprintf(stderr, "dl_check: layer %d: ca_biases to biases matrix mismatch: %d height to %d height.\n", i, this->ca_biases.height, this->biases.height);
            }
            if (this->ca_biases.width != this->biases.width)
            {
                fprintf(stderr, "dl_check: layer %d: ca_biases to biases matrix mismatch: %d width to %d width.\n", i, this->ca_biases.width, this->biases.width);
            }
            if (this->der_last_activations.height != this->last_activations.height)
            {
                fprintf(stderr, "dl_check: layer %d: last_activations and der_last_activations different: %d height to %d height.\n", i, this->last_activations.height, this->der_last_activations.height);
            }

            return 0;
        }
        else
        {
            this = next;
            next = this->next;
            i++;
        }
    }

    return i + 1;
}

// Assembles all the nodes into a linked structure usable by dl_process and dl_check using the head node
// Calls dl_check to check for the result
//  0 -> not valid
//  number of layers -> valid
int dl_assemble(node *head, node **hidden_layers, int hidden_count, node *tail)
{
    head->prev = NULL;
    tail->next = NULL;
    if (hidden_count == 0)
    {
        head->next = tail;
        tail->prev = head;
    }
    else
        for (int i = 0; i < hidden_count; i++)
        {
            // assign previous layer ->next and current layer ->prev
            if (i == 0)
            {
                head->next = hidden_layers[i];
                hidden_layers[i]->prev = head;
            }
            else
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
    matrix_free(head->der_last_activations);

    if (head->next) // if not tail
    {
        // free subsequent layers
        dl_free(head->next);
    }

    // if allocated
    if (head->self)
        free(head->self);
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
    while ((aux = aux->next))
        node_count++;
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
node *dl_load(const char *filename)
{
    unsigned int node_count;
    FILE *fd = fopen(filename, "rb");

    fread(&node_count, sizeof(unsigned int), 1, fd);

    node *nodes = (node *)malloc(sizeof(node) * node_count);
    int h, w;
    for (int i = 0; i < node_count; i++)
    {
        // Since only one malloc is used, it only needs to be freed once for the input node
        nodes[i].self = NULL;

        nodes[i].n_ca = 0;

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
        nodes[i].der_last_activations = matrix_create(h, w);
        for (int r = 0; r < h; r++)
            fread(nodes[i].biases.matrix[r], sizeof(double), w, fd);
    }

    node **hidden = (node **)malloc(sizeof(node *) * (node_count - 2));
    for (int i = 0; i < node_count - 2; i++)
    {
        hidden[i] = &nodes[i + 1];
    }
    dl_assemble(&nodes[0], hidden, node_count - 2, &nodes[node_count - 1]);
    free(hidden);

    fclose(fd);
    nodes[0].self = nodes;
    return &nodes[0];
}

// Calculates the cost for the result
// Cost = Sum((result - expected)^2)
double dl_cost(matrix result, matrix expected)
{
    if (result.height != expected.height)
    {
        fprintf(stderr, "dl_cost: result and expected matrix heights differ: %d to %d.\n", result.height, expected.height);
        return 0;
    }

    double cost = 0;
    double difference;

    for (int i = 0; i < result.height; i++)
    {
        difference = result.matrix[i][0] - expected.matrix[i][0];
        cost += difference * difference;
    }

    return cost;
}

// Applies the stored adjustments to the weights and biases
void dl_adjust(node *head)
{
    if (head == NULL)
        return;

    if (head->prev && head->n_ca) // not the input layer and adjustments to make
    {
        matrix_sub(head->weights, matrix_scalar_mult(head->ca_weights, (double) 1 / head->n_ca));
        matrix_sub(head->biases, matrix_scalar_mult(head->ca_biases, (double) 1 / head->n_ca));
        // matrix_sub(head->weights, head->ca_weights);
        // matrix_sub(head->biases, head->ca_biases);
        matrix_zero(head->ca_weights);
        matrix_zero(head->ca_biases);
        head->n_ca = 0;
    }
    // check the network once when at the input layer
    else if (!dl_check(head))
    {
        fprintf(stderr, "dl_adjust: Adjustment matrices incompatible\n");
        return;
    }
    dl_adjust(head->next);
}

matrix dl_log_loss(matrix output, matrix expected)
{
    matrix loss = matrix_create(output.height, 1);
    for (int i = 0; i < output.height; i++)
    {
        loss.matrix[i][0] = - output.matrix[i][0] * log(1e-15 + expected.matrix[i][0]);
    }
    
    return loss;
}

matrix dl_mse_loss(matrix output, matrix expected)
{
    matrix loss = matrix_create(output.height, 1);
    for (int i = 0; i < output.height; i++)
    {
        loss.matrix[i][0] = 2 * (output.matrix[i][0] - expected.matrix[i][0]);
    }
    
    return loss;
}

// Calculates all adjustments for a layer and recursively propagates to the previous node until the input node is reached
void dl_backpropagate(node *n, matrix loss, double alpha)
{
    // check if input layer
    if (!n->prev)
        return;
    
    matrix new_loss = matrix_create(n->prev->last_activations.height, 1);
    for (int i = 0; i < n->ca_weights.height; i++)
    {
        for (int j = 0; j < n->ca_weights.width; j++)
        {
            // strictly speaking, the loss should be multiplied by 2, but that is a constant that can be part of alpha
            n->ca_weights.matrix[i][j] += n->der_last_activations.matrix[i][0]
                                        * loss.matrix[i][0]
                                        * n->prev->last_activations.matrix[j][0]
                                        * alpha;
        }
        n->ca_biases.matrix[i][0] += loss.matrix[i][0] * alpha;
    }

    for (int j = 0; j < new_loss.height; j++)
    {
        for (int i = 0; i < n->ca_weights.height; i++)
        {
            new_loss.matrix[j][0] += n->der_last_activations.matrix[i][0] * loss.matrix[i][0] * n->weights.matrix[i][j];
        }
    }

    // backpropagate
    n->n_ca++;
    dl_backpropagate(n->prev, new_loss, alpha);
    matrix_free(new_loss);
}

// Starts the backwards pass and calculates adjustments to weights and biases for all layers but the input
// expected is the expected output
// alpha is the learning constant
void dl_backwards_pass(node *head, matrix expected, double alpha)
{
    // look for tail
    node *tail = head;
    while (tail->next)
        tail = tail->next;

    if (tail->last_activations.height != expected.height)
    {
        fprintf(stderr, "dl_backwards_pass: output and expected matrix height differ: %d to %d.\n", tail->last_activations.height, expected.height);
        exit(1);
    }

    // start backwards pass at tail
    matrix loss = dl_mse_loss(tail->last_activations, expected);
    dl_backpropagate(tail, loss, alpha);
    matrix_free(loss);
}