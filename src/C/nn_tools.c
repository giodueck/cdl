#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <string.h>

#include "nn_tools.h"

#define NULL_MATRIX (matrix) { .height = 0, .width = 0, .matrix = NULL }

// Returns 0 matrix struct with width number of columns and height number of rows
matrix matrix_create(int height, int width)
{
    matrix A;
    A.height = height;
    A.width = width;
    A.matrix = NULL;

    A.matrix = (double **) malloc(sizeof(double *) * A.height);
    if (!A.matrix)
    {
        fprintf(stderr, "matric_create: Allocation error\n");
        A.height = 0;
        A.width = 0;
        A.matrix = NULL;
        return A;
    }

    A.matrix[0] = (double *) malloc(sizeof(double) * A.height * A.width);
    if (!A.matrix)
    {
        fprintf(stderr, "matric_create: Allocation error\n");
        free(A.matrix);
        A.height = 0;
        A.width = 0;
        A.matrix = NULL;
        return A;
    }
    for (int i = 1; i < A.height; i++)
    {
        A.matrix[i] = A.matrix[0] + i * A.width;
    }

    matrix_zero(A);
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

    free(A.matrix[0]);
    free(A.matrix);
    A.matrix = NULL;
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
// mat[0] should point to a contiguous block of memory
matrix matrix_init(matrix A, double **mat)
{
    memcpy(A.matrix[0], mat[0], sizeof(double) * A.height * A.width);

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
    memset(A.matrix[0], 0, sizeof(double) * A.height * A.width);

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

// Checks if the network is valid
//  0 -> not valid
//  number of layers -> valid
int dl_check(node *in_node)
{
    node *current, *next;

    current = in_node;
    next = in_node->next;

    int i = 0;

    while (next != NULL)
    {
        if (next->weights.width != current->weights.height
        || current->biases.height != current->weights.height
        || current->biases.width != 1
        || current->biases.height != current->last_activations.height
        || current->ca_weights.height != current->weights.height
        || current->ca_weights.width != current->weights.width
        || current->ca_biases.height != current->biases.height
        || current->ca_biases.width != current->biases.width
        || current->der_last_activations.height != current->last_activations.height)
        {
            if (next->weights.width != current->weights.height)
            {
                fprintf(stderr, "dl_check: Layer %d: weights matrix mismatch: %d height to %d width.\n", i, current->weights.height, next->weights.width);
            }
            if (current->biases.height != current->weights.height)
            {
                fprintf(stderr, "dl_check: Layer %d: biases to weights matrix mismatch: %d height to %d height.\n", i, current->biases.height, current->weights.height);
            }
            if (current->biases.width != 1)
            {
                fprintf(stderr, "dl_check: Layer %d: biases matrix too wide: %d.\n", i, current->biases.width);
            }
            if (current->biases.height != current->last_activations.height)
            {
                fprintf(stderr, "dl_check: Layer %d: biases to last_activations matrix mismatch: %d height to %d height.\n", i, current->biases.height, current->last_activations.height);
            }
            if (current->ca_weights.height != current->weights.height)
            {
                fprintf(stderr, "dl_check: Layer %d: ca_weights to weights matrix mismatch: %d height to %d height.\n", i, current->ca_weights.height, current->weights.height);
            }
            if (current->ca_weights.width != current->weights.width)
            {
                fprintf(stderr, "dl_check: Layer %d: ca_weights to weights matrix mismatch: %d width to %d width.\n", i, current->ca_weights.width, current->weights.width);
            }
            if (current->ca_biases.height != current->biases.height)
            {
                fprintf(stderr, "dl_check: Layer %d: ca_biases to biases matrix mismatch: %d height to %d height.\n", i, current->ca_biases.height, current->biases.height);
            }
            if (current->ca_biases.width != current->biases.width)
            {
                fprintf(stderr, "dl_check: Layer %d: ca_biases to biases matrix mismatch: %d width to %d width.\n", i, current->ca_biases.width, current->biases.width);
            }
            if (current->der_last_activations.height != current->last_activations.height)
            {
                fprintf(stderr, "dl_check: Layer %d: last_activations and der_last_activations different: %d height to %d height.\n", i, current->last_activations.height, current->der_last_activations.height);
            }

            return 0;
        }
        else
        {
            current = next;
            next = current->next;
            i++;
        }
    }

    return i + 1;
}

// Prints the net's structure to the console
void dl_print_structure(node *head)
{
    printf("Structure: %d", head->biases.height);
    
    while ((head = head->next)) // go to next layer until it is null
    {
        printf("-%d", head->biases.height);
    }
    printf("\n");
}

// Prints structure into buf up to len - 1
// Returns number of characters written
int dl_structure_str(node *head, char *buf, int len)
{
    char auxbuf[BUFSIZ];
    sprintf(auxbuf, "%d", head->biases.height);
    while ((head = head->next)) // go to next layer until it is null
    {
        snprintf(auxbuf, BUFSIZ, "%s-%d", auxbuf, head->biases.height);
    }

    int i;
    for (i = 0; i < len; i++)
    {
        buf[i] = auxbuf[i];
        if (i == len - 1)
        {
            buf[i] = '\0';
            break;
        }
        if (buf[i] == '\0')
            break;
    }
    return i;
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
    const char* magic = "CDLD";
    FILE *fd = fopen(filename, "wb");
    fwrite(magic, 1, 4, fd);

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
// If the magic number "CDLD" is not present, the return is NULL
node *dl_load(const char *filename)
{
    unsigned int node_count;
    char magic[5];
    FILE *fd = fopen(filename, "rb");
    if (!fd)
    {
        perror("dl_load");
        return NULL;
    }
    fread(magic, 1, 4, fd);
    magic[4] = '\0';
    if (strcmp(magic, "CDLD") != 0)
    {
        fprintf(stderr, "dl_load: File '%s' not in the right format.\n", filename);
        fclose(fd);
        return NULL;
    }

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

// Create a copy of a Deep Neural Network
node *dl_copy(node *head)
{
    int n_inputs = 0, n_layers = 0;
    int *sizes, sizes_len = 0;
    node *copy, *og_copy, *og_head = head;

    sizes = malloc(sizeof(int) * (sizes_len += 10));
    
    while(head)
    {
        if (head->prev == NULL)
            n_inputs = head->last_activations.height;
        else
        {
            if (sizes_len <= n_layers + 1)
                sizes = realloc(sizes, sizeof(int) * (sizes_len += 10));
            sizes[n_layers++] = head->last_activations.height;
        }
        head = head->next;
    }

    copy = dl_create(n_inputs, n_layers, sizes);
    og_copy = copy;
    free(sizes);

    head = og_head;
    while(head)
    {
        matrix_init(copy->weights, head->weights.matrix);
        matrix_init(copy->biases, head->biases.matrix);
        head = head->next;
        copy = copy->next;
    }
    return og_copy;
}

// Using an input column and the neural networks input node, calculate the result column
// Does not use recursion to be able to use GPU acceleration
matrix dl_process(node *in_node, matrix input)
{
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

    matrix act;
    node *tail = in_node;
    matrix_init(in_node->last_activations, input.matrix);
    in_node = in_node->next;
    while(in_node)
    {
        act = matrix_add(matrix_mult(in_node->weights, in_node->prev->last_activations), in_node->biases);
        matrix_init(in_node->last_activations, act.matrix);
        matrix_init(in_node->der_last_activations, act.matrix);
        matrix_free(act);

        // act is the vector of neuron activations for this layer,
        // for the head it is the input, for any other layer, the result of this calculation
        matrix_sigmoid(in_node->last_activations);

        // der_act, derivated activations, is used for backpropagation and not needed for the input layer
        matrix_derivated_sigmoid(in_node->der_last_activations);

        // go to next layer
        tail = in_node;
        in_node = in_node->next;
    }

    // input will be the result, not the original input
    return matrix_copy(tail->last_activations);
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

// Compute loss matrix, 2 * (a(i) - y(i))
matrix dl_mse_loss(matrix output, matrix expected)
{
    matrix loss = matrix_create(output.height, 1);
    for (int i = 0; i < output.height; i++)
    {
        loss.matrix[i][0] = 2 * (output.matrix[i][0] - expected.matrix[i][0]);
    }
    
    return loss;
}

// Starts the backwards pass and calculates adjustments to weights and biases for all layers but the input
// expected is the expected output
// alpha is the learning constant
void dl_backpropagation(node *head, matrix loss, double alpha)
{
    // look for tail
    node *tail = head;
    while (tail->next)
        tail = tail->next;
    
    if (tail->last_activations.height != loss.height)
    {
        fprintf(stderr, "dl_backwards_pass: Output and loss matrix height differ: %d to %d.\n", tail->last_activations.height, loss.height);
        exit(1);
    }

    // start backwards pass at tail, stop when the input layer has been reached
    matrix new_loss;
    loss = matrix_copy(loss);
    while(tail->prev)
    {
        new_loss = matrix_create(tail->prev->last_activations.height, 1);

        for (int i = 0; i < tail->ca_weights.height; i++)
        {
            for (int j = 0; j < tail->ca_weights.width; j++)
            {
                // strictly speaking, the loss should be multiplied by 2, but that is a constant that can be part of alpha
                tail->ca_weights.matrix[i][j] += tail->der_last_activations.matrix[i][0]
                                            * loss.matrix[i][0]
                                            * tail->prev->last_activations.matrix[j][0]
                                            * alpha;
            }
            tail->ca_biases.matrix[i][0] += loss.matrix[i][0] * alpha;
        }

        for (int j = 0; j < new_loss.height; j++)
        {
            for (int i = 0; i < tail->ca_weights.height; i++)
            {
                new_loss.matrix[j][0] += tail->der_last_activations.matrix[i][0] * loss.matrix[i][0] * tail->weights.matrix[i][j];
            }
        }

        // backpropagate
        tail->n_ca++;
        matrix_free(loss);
        loss = new_loss;
        tail = tail->prev;
    }
    matrix_free(loss);
}

void cart_to_polar(double x, double y, double *r, double *phi)
{
    *r = sqrt(x*x + y*y);
    *phi = atan2(y, x);
}

void polar_to_cart(double r, double phi, double *x, double *y)
{
    *x = r * cos(phi);
    *y = r * sin(phi);
}

void place_aliased_pixel(uint8_t *image, double x, double y, int value)
{
    if (value == 0)
        return;

    int x_ = (int)x;
    int y_ = (int)y;

    double xmod = x - x_;
    double ymod = y - y_;
    double xmodinv = 1 - xmod;
    double ymodinv = 1 - ymod;

    // Consider if pixel is in image and the pixel is not overflowing
    if (x_ >= 0 && y_ >= 0 && x_ < 28 && y_ < 28)
        image[y_ * 28 + x_] = (image[y_ * 28 + x_] + value * xmodinv * ymodinv > 255) ? 255 : image[y_ * 28 + x_] + value * xmodinv * ymodinv ;
    if (x_ + 1 >= 0 && y_ >= 0 && x_ + 1 < 28 && y_ < 28)
        image[y_ * 28 + x_ + 1] = (image[y_ * 28 + x_ + 1] + value * xmod * ymodinv > 255) ? 255 : image[y_ * 28 + x_ + 1] + value * xmod * ymodinv ;
    if (x_ >= 0 && y_ + 1 >= 0 && x_ < 28 && y_ + 1 < 28)
        image[(y_ + 1) * 28 + x_] = (image[(y_ + 1) * 28 + x_] + value * xmodinv * ymod > 255) ? 255 : image[(y_ + 1) * 28 + x_] + value * xmodinv * ymod ;
    if (x_ + 1 >= 0 && y_ + 1 >= 0 && x_ + 1 < 28 && y_ + 1 < 28)
        image[(y_ + 1) * 28 + x_ + 1] = (image[(y_ + 1) * 28 + x_ + 1] + value * xmod * ymod > 255) ? 255 : image[(y_ + 1) * 28 + x_ + 1] + value * xmod * ymod ;
}

// Randomly rotate the image to create a new one
void dl_rotate_image_rand(uint8_t *dst, uint8_t *image)
{
    int x, y;
    double r, phi, x_, y_;
    memset(dst, 0, sizeof(uint8_t) * 784);
    double rot = randfrom(-DL_ROTATIONMAX, +DL_ROTATIONMAX);

    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            // translate center of image to origin
            x = j - 14;
            y = i - 14;
            cart_to_polar(x, y, &r, &phi);
            phi += rot;
            polar_to_cart(r, phi, &x_, &y_);
            // translate back to center of image
            x_ += 14;
            y_ += 14;
            // set new pixel
            place_aliased_pixel(dst, x_, y_, image[i * 28 + j]);
        }
    }
}

// Randomly shift the image vertically and horizontally to create a new one
void dl_shift_image_rand(uint8_t *dst, uint8_t *image)
{
    memset(dst, 0, sizeof(uint8_t) * 784);
    int di = rand() % (2 * DL_SHIFTMAX) - DL_SHIFTMAX;
    int dj = rand() % (2 * DL_SHIFTMAX) - DL_SHIFTMAX;
    int is, js;

    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            is = i + di;
            js = j + dj;
            if (is * 28 + js >= 0 && is * 28 + js < 784)
                dst[is * 28 + js] = image[i * 28 + j];
        }
    }
}

// Randomly scale the image up or down to create a new one
// uint8_t *dl_zoom_image_rand(uint8_t *image)
// {
    
// }

// Randomly apply shear to the image to create a new one
void dl_shear_image_rand(uint8_t *dst, uint8_t *image)
{
    memset(dst, 0, sizeof(uint8_t) * 784);
    double angle = randfrom(0, 2 * M_PI);
    double shear_val = randfrom(0, DL_SHEARMAX);
    double x_, y_;

    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            x_ = j + i * cos(angle) * shear_val;
            y_ = i + j * sin(angle) * shear_val;
            place_aliased_pixel(dst, x_, y_, image[i * 28 + j]);
        }
    }
}