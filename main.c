#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn_tools.h"
#include "swap.h"
#include <errno.h>

int get_len(const char *filename)
{
    int32_t magic, len;

    FILE *fd = fopen(filename, "rb");
    if (fd == NULL)
    {
        perror(filename);
        exit(1);
    }

    fread(&magic, sizeof(int32_t), 1, fd);
    fread(&len, sizeof(int32_t), 1, fd);
    if (magic != 2049 && magic != 2051)
    {
        magic = swap_int32(magic);
        len = swap_int32(len);
    }

    fclose(fd);

    return len;
}

// Returns malloced array of uint8_t, or unsigned char
uint8_t *get_labels(const char *filename)
{
    int32_t magic, len;
    uint8_t *res = NULL;

    FILE *fd = fopen(filename, "rb");
    if (fd == NULL)
    {
        perror(filename);
        exit(1);
    }

    fread(&magic, sizeof(int32_t), 1, fd);
    fread(&len, sizeof(int32_t), 1, fd);
    if (magic != 2049)
    {
        magic = swap_int32(magic);
        len = swap_int32(len);
    }

    res = (uint8_t *)malloc(sizeof(uint8_t) * len);
    if (res == NULL)
    {
        perror("");
        exit(1);
    }

    for (int i = 0; i < len; i++)
    {
        fread(res + i, sizeof(uint8_t), 1, fd);
    }

    fclose(fd);

    return res;
}

// Returns malloced array of pixels as uint8_t, or unsigned char
// Every 784 (28*28) pixels is one image
uint8_t *get_images(const char *filename)
{
    int32_t magic, len, h, w;
    uint8_t *res = NULL;

    FILE *fd = fopen(filename, "rb");
    if (fd == NULL)
    {
        perror(filename);
        exit(1);
    }

    fread(&magic, sizeof(int32_t), 1, fd);
    fread(&len, sizeof(int32_t), 1, fd);
    fread(&h, sizeof(int32_t), 1, fd);
    fread(&w, sizeof(int32_t), 1, fd);
    if (magic != 2049)
    {
        magic = swap_int32(magic);
        len = swap_int32(len);
        h = swap_int32(h);
        w = swap_int32(w);
    }

    res = (uint8_t*) malloc(sizeof(uint8_t) * len * h * w);
    if (res == NULL)
    {
        perror("");
        exit(1);
    }

    for (int i = 0; i < len; i++)
    {
        fread(res + i * h * w, sizeof(uint8_t), h * w, fd);
    }

    fclose(fd);

    return res;
}

// Returns a number between 0 and 1
// Values under min are clamped to 0, values over max, to 1
double lerp(int n, int min, int max)
{
    if (n <= min)
        return 0;
    if (n >= max)
        return 1;

    return n / (double)(max - min);
}

void print_label(uint8_t *labels, int index)
{
    printf("Label: %d\n", labels[index]);
}

void print_image(uint8_t **images_v, uint8_t *labels, int count, int index)
{
    while (index < 0)
    {
        index += count;
    }
    
    while (index >= count)
    {
        index -= count;
    }

    const char *shading = " .:$#";
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            // putchar(shading[(int) (4 * lerp(images[k * 784 + i * 28 + j], 0, 255))]);
            putchar(shading[(int) (4 * lerp(images_v[index][i * 28 + j], 0, 255))]);
            putchar(' ');
        }
        putchar('\n');
    }
}

void print_image_smaller(uint8_t **images_v, uint8_t *labels, int count, int index, int downscaling)
{
    if (downscaling <= 1)
    {
        print_image(images_v, labels, count, index);
        return;
    }

    while (index < 0)
    {
        index += count;
    }
    
    while (index >= count)
    {
        index -= count;
    }

    const char *shading = " .:$#";
    for (int i = 0; i < 28; i += downscaling)
    {
        for (int j = 0; j < 28; j += downscaling)
        {
            // putchar(shading[(int) (4 * lerp(images[k * 784 + i * 28 + j], 0, 255))]);
            putchar(shading[(int) (4 * lerp(images_v[index][i * 28 + j], 0, 255))]);
            putchar(' ');
        }
        putchar('\n');
    }
}

// Shuffles the images and labels vectors
void shuffle_images(uint8_t **images_v, uint8_t *labels, int len)
{
    uint8_t *x, y;
    int j;
    for (int i = 0; i < len; i++)
    {
        x = images_v[i];
        y = labels[i];
        j = rand() % (len - i) + i;
        images_v[i] = images_v[j];
        labels[i] = labels[j];
        images_v[j] = x;
        labels[j] = y;
    }
}

int main()
{
    // Load training data
    uint8_t *labels = get_labels("mnist/train-labels.idx1-ubyte");
    int count = get_len("mnist/train-labels.idx1-ubyte");
    uint8_t *images = get_images("mnist/train-images.idx3-ubyte");
    uint8_t **images_v = (uint8_t**) malloc(sizeof(uint8_t*) * count);

    // Every *images_v points to one array of 784 pixels, or one image
    for (int i = 0; i < count; i++)
    {
        images_v[i] = &images[i * 784];
    }
    srand(time(0));
    // shuffle_images(images_v, labels, count);

    // Create Neural Network
    node head, l1, l2, tail;
    node *hidden_layers[2];
    int n_inputs = 784;
    int l1_size = 24;
    int l2_size = 24;
    int n_outputs = 10;

    int randmin = -1, randmax = 1;

    // Net structure creation
    int method = 4;

    // Method 1: create nodes manually, then assemble into a net
    // These will be 0, but for consistency are added anyways
    if (method == 1)
    {
        head.weights = matrix_create(n_inputs, 1);
        head.ca_weights = matrix_create(n_inputs, 1);
        head.biases = matrix_create(n_inputs, 1);
        head.ca_biases = matrix_create(n_inputs, 1);
        head.last_activations = matrix_create(n_inputs, 1);
        head.self = NULL;

        l1.weights = matrix_init_rand(matrix_create(l1_size, n_inputs), randmin, randmax);
        l1.ca_weights = matrix_create(l1_size, n_inputs);
        l1.biases = matrix_init_rand(matrix_create(l1_size, 1), randmin, randmax);
        l1.ca_biases = matrix_create(l1_size, 1);
        l1.last_activations = matrix_create(l1_size, 1);
        l1.self = NULL;
        hidden_layers[0] = &l1;

        l2.weights = matrix_init_rand(matrix_create(l2_size, l1_size), randmin, randmax);
        l2.ca_weights = matrix_create(l2_size, l1_size);
        l2.biases = matrix_init_rand(matrix_create(l2_size, 1), randmin, randmax);
        l2.ca_biases = matrix_create(l2_size, 1);
        l2.last_activations = matrix_create(l2_size, 1);
        l2.self = NULL;
        hidden_layers[1] = &l2;

        tail.weights = matrix_init_rand(matrix_create(n_outputs, l2_size), randmin, randmax);
        tail.ca_weights = matrix_create(n_outputs, l2_size);
        tail.biases = matrix_init_rand(matrix_create(n_outputs, 1), randmin, randmax);
        tail.ca_biases = matrix_create(n_outputs, 1);
        tail.last_activations = matrix_create(n_outputs, 1);
        tail.self = NULL;

        dl_assemble(&head, hidden_layers, 2, &tail);
    }

    // Method 2: load nodes from file and assemble into a net
    else if (method == 2)
    {
        head = dl_load("test.dld");
    }

    // Method 3: create nodes in sequence, assembling implicitly
    else if (method == 3)
    {
        // Note: assigning a pointed-to value to a variable and not storing the pointer is dumb,
        // because the value's actual location can't be accessed. Luckily an allocated node
        // points to its own location for freeing, but all fixed size variables are lost, e.g.
        // .next and .prev
        node *head_, *l1_, *l2_, *tail_;

        head_ = dl_create_node(DL_INPUT, n_inputs, NULL);
        l1_ = dl_create_node(DL_HIDDEN, l1_size, head_->self);
        l2_ = dl_create_node(DL_HIDDEN, l2_size, l1_->self);
        tail_ = dl_create_node(DL_OUTPUT, n_outputs, l2_->self);
        
        if (dl_check(head_) < 4)
        {
            fprintf(stderr, "Method 3: %d layers found, %d expected.\n", dl_check(head_), 4);
            return 0;
        }

        // This means all fixed size data, which will not be modified, is copied over to the variable head.
        // Variable size data still is pointed to, and will be accessible without issues
        head = *head_;
    }

    else if (method == 4)
    {
        int sizes[3] = { l1_size, l2_size, n_outputs };
        head = *dl_create(784, 3, sizes);

        if (dl_check(&head) < 4)
        {
            fprintf(stderr, "Method 3: %d layers found, %d expected.\n", dl_check(&head), 4);
            return 0;
        }
    }

    // Create input vector from an image
    matrix input = matrix_create(n_inputs, 1);
    matrix output;
    matrix expected = matrix_create(10, 0);
    int img_index = 0;
    print_image(images_v, labels, count, img_index);
    print_image_smaller(images_v, labels, count, img_index, 2);
    print_label(labels, img_index);

    output = dl_process(&head, input);
    if (!output.matrix) exit(1);
    int choice;
    double confidence = 0, uncertainty = 0;
    for (int i = 0; i < n_outputs; i++)
    {
        if (output.matrix[i][0] > confidence)
        {
            confidence = output.matrix[i][0];
            choice = i;
        }
    }
    for (int i = 0; i < n_outputs; i++)
    {
        if (output.matrix[i][0] > uncertainty && output.matrix[i][0] < confidence)
            uncertainty = output.matrix[i][0];
    }
    
    matrix_print(output, NULL);
    printf("Choice: %d\nConfidence: %.2lf%%\nLabel: %d\n", choice, (confidence - uncertainty) * 100.0, labels[img_index]);

    expected.matrix[labels[img_index]][0] = 1;
    printf("Cost: %.4lf\n", dl_cost(output, expected));

    dl_dump(&head, "test.dld");

    dl_adjust(&head);
    // // Print activations
    // printf("1st layer activations:\n");
    // matrix_print(head.next->last_activations, "\n");
    // printf("2nd layer activations:\n");
    // matrix_print(head.next->next->last_activations, "\n");
    // printf("Output layer activations:\n");
    // matrix_print(head.next->next->next->last_activations, "\n");

    // Cleanup code
    matrix_free(input);
    matrix_free(output);
    matrix_free(expected);

    dl_free(&head);

    free(labels);
    free(images);
    free(images_v);
    return 0;
}