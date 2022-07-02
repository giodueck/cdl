#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include "nn_tools.h"
#include "swap.h"
#include "getopt.h"

// range of values in the dataset
const int minval = 0, maxval = 255;

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

void print_image(uint8_t **images_v, int count, int index)
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
            putchar(shading[(int) (4 * lerp(images_v[index][i * 28 + j], minval, maxval))]);
            putchar(' ');
        }
        putchar('\n');
    }
}

// Only prints every NxNth pixel, downscaling defines that N
void print_image_smaller(uint8_t **images_v, int count, int index, int downscaling)
{
    if (downscaling <= 1)
    {
        print_image(images_v, count, index);
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
            putchar(shading[(int) (4 * lerp(images_v[index][i * 28 + j], minval, maxval))]);
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

uint8_t *augment_labels(uint8_t *labels, int count, int factor)
{
    uint8_t *aug_labels = (uint8_t*) malloc(sizeof(uint8_t) * count * factor);

    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < factor; j++)
        {
            aug_labels[i * factor + j] = labels[i];
        }
    }
    
    return aug_labels;
}

uint8_t *augment_images(uint8_t *images, int count, int factor)
{
    uint8_t *aug_images = (uint8_t*) malloc(sizeof(uint8_t*) * 784 * count * factor);
    uint8_t *im0, *im1, *im2;

    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < factor; j++)
        {
            if (j > 0)
            {
                im0 = dl_rotate_image_rand(&images[i * 784]);
                im1 = dl_shift_image_rand(im0);
                im2 = dl_shear_image_rand(im1);
                memcpy(&aug_images[(i * factor + j) * 784], im2, sizeof(uint8_t) * 784);
                free(im0);
                free(im1);
                free(im2);
            } else
                memcpy(&aug_images[(i * factor + j) * 784], &images[i * 784], sizeof(uint8_t) * 784);
        }
    }
    
    return aug_images;
}

void test_model(node *head)
{
    // Load testing data
    uint8_t *t_labels = get_labels("mnist/t10k-labels.idx1-ubyte");
    int t_count = get_len("mnist/t10k-labels.idx1-ubyte");
    uint8_t *t_images = get_images("mnist/t10k-images.idx3-ubyte");
    uint8_t **t_images_v = (uint8_t**) malloc(sizeof(uint8_t*) * t_count);

    for (int i = 0; i < t_count; i++)
        t_images_v[i] = &t_images[i * 784];

    int n_inputs = head->biases.height;
    node *aux = head;
    while (aux->next)
        aux = aux->next;
    int n_outputs = aux->biases.height;
    
    // Testing against new data
    printf("Testing with new data...");
    fflush(stdout);
    double avg_cost = 0;
    int correct_answers = 0;
    double confidence;
    int choice;
    matrix input = matrix_create(n_inputs, 1);
    matrix expected = matrix_create(n_outputs, 1);
    matrix output;
    for (int i = 0; i < t_count; i++)
    {
        confidence = 0;

        // input creation
        for (int k = 0; k < n_inputs; k++)
        {
            input.matrix[k][0] = (double) t_images_v[i][k];
        }

        expected = matrix_zero(expected);
        expected.matrix[t_labels[i]][0] = 1;

        // processing
        output = dl_process(head, input);

        // verification
        for (int j = 0; j < n_outputs; j++)
        {
            if (output.matrix[j][0] > confidence)
            {
                confidence = output.matrix[j][0];
                choice = j;
            }
        }
        
        // counting
        correct_answers += (choice == t_labels[i]);
        avg_cost += dl_cost(output, expected);

        matrix_free(output);
    }
    avg_cost /= t_count;
    printf("\rCorrect answers: %d/%d = %.2f%%\tAvg cost: %.4lf\n", correct_answers, t_count, (double) correct_answers / t_count * 100.0, avg_cost);

    matrix_free(input);
    matrix_free(expected);
    matrix_free(output);
    free(t_labels);
    free(t_images);
    free(t_images_v);
}

void help(char **argv)
{
    const char *msg =
"Usage:  %s [-l <size>] [-f <filename>] [-t <filename>]\n\
        [-a <rate>] [-r <count>] [-b <count>] [-h] [-g <factor>]\n\
\n\
        -a <rate>       Specify a learning rate. Default is 0.15\n\
        -b <count>      Specify the size of each batch of training images used in stochastic\n\
                        gradient descent. Default is 100\n\
        -f <filename>   Specify the filename of the resulting model file. The '.cdl' extension\n\
                        is added automatically.\n\
        -l <size>       Can be used multiple times. Specify the size of the next hidden layer.\n\
                        Each layer is added in order of appearance.\n\
        -e <count>      Specify the number of training rounds (epochs) over the shuffled training set.\n\
                        Default is 10.\n\
        -h              Show this help menu.\n\
        -g <factor>     Specify how many times to augment the data. Default is 5. To disable use\n\
                        -g 1.\n\
\n\
Example: %s -l 100 -l 50 -a 0.2 -r 15 -f myDLModel\n";

    printf(msg, argv[0], argv[0]);
}

int main(int argc, char **argv)
{
    // Handle args
    int n_layers = 1;
    // reserve one space for output layer
    int *sizes = (int*) malloc(sizeof(int));
    const int n_inputs = 784;
    const int n_outputs = 10;
    char filename[FILENAME_MAX];
    sprintf(filename, "dl-%ld.dld", time(0));
    // range for initial values
    const int randmin = -1, randmax = 1;
    // learning rate
    double alpha = 0.15;
    // Train the model several times over shuffled versions of the same dataset
    int epochs = 10;
    // Train in batches of random images
    int batch_size = 100;
    // Augment data
    int augmentation_factor = 5;
    
    int c;
    extern int optopt;
    extern char *optarg;
    extern int opterr;
    opterr = 0; // to suppress getopt error messages

    char testflag = 0;

    /*
    Files: invent some magic number to put in front of the file, so I can identify valid files
    C  D  L  D
    67 68 76 67

    Args:
        Implemented:
            -l <size>: specify hidden layer size, can specify multiple in order
            -f <filename>: specify file name to save model to
            -t <filename>: specify file to test model
            -a: specify the learning rate
            -e: specify how many training runs (epochs) to do
            -b: specify batch size
            -h: help menu
            -g: specify the factor for data augmentation

        Not implemented:
    */

    while ((c = getopt(argc, argv, "l:f:t:a:e:b:hg:")) != -1)
    {
        switch (c)
        {
        case 'l':
            // store hidden layer and expand sizes vector to store output layer
            sizes[n_layers - 1] = atoi(optarg);
            sizes = (int*) realloc(sizes, sizeof(int) * ++n_layers);
            break;
        case 'f':
            strcpy(filename, optarg);
            strcat(filename, ".dld");
            break;
        case 't':
            testflag = 1;
            strcpy(filename, optarg);
            break;
        case 'a':
            alpha = atof(optarg);
            if (!alpha)
            {
                fprintf(stderr, "Option -%c requires a non-zero decimal argument\n", optopt);
                help(argv);
                return 1;
            }
            break;
        case 'e':
            epochs = abs(atoi(optarg));
            if (!epochs)
            {
                fprintf(stderr, "Option -%c requires a non-zero integer argument\n", optopt);
                help(argv);
                return 1;
            }
            break;
        case 'b':
            batch_size = abs(atoi(optarg));
            if (!batch_size)
            {
                fprintf(stderr, "Option -%c requires a non-zero integer argument\n", optopt);
                help(argv);
                return 1;
            }
            break;
        case 'h':
            help(argv);
            return 0;
        case 'g':
            augmentation_factor = abs(atoi(optarg));
            if (!augmentation_factor)
            {
                fprintf(stderr, "Option -%c requires a non-zero integer argument\n", optopt);
                help(argv);
                return 1;
            }
            break;
        case ':':
            fprintf(stderr, "Option -%c requires an argument\n", optopt);
            return 1;
        case '?':
            if (isprint(c))
                fprintf(stderr, "Unknown option '-%c'\n", optopt);
            else
                fprintf(stderr, "Unknown option character '\\x%x'.\n", optopt);
            
            help(argv);
            return 1;
        default:
            printf("%d %c\n", c, optopt);
            abort();
        }
    }

    // Only test a model
    if (testflag)
    {
        free(sizes);
        node *head = dl_load(filename);
        if (head)
        {
            dl_print_structure(head);
            test_model(head);
            dl_dump(head, filename);
            dl_free(head);
        }

        return 0;
    }

    // no arguments given, default to 2 16-neuron hidden layers
    if (argc == 1)
    {
        sizes[n_layers - 1] = 16;
        sizes = (int*) realloc(sizes, sizeof(int) * ++n_layers);
        sizes[n_layers - 1] = 16;
        sizes = (int*) realloc(sizes, sizeof(int) * ++n_layers);
    }

    // last layer will be output
    sizes[n_layers - 1] = n_outputs;

    // Load training data
    printf("Loading training data...");
    fflush(stdout);
    int training_count = get_len("mnist/train-labels.idx1-ubyte");
    uint8_t *training_labels = get_labels("mnist/train-labels.idx1-ubyte");
    uint8_t *training_images = get_images("mnist/train-images.idx3-ubyte");
    uint8_t **training_images_v = (uint8_t**) malloc(sizeof(uint8_t*) * training_count);

    // Every *images_v points to one array of 784 pixels, or one image
    for (int i = 0; i < training_count; i++)
        training_images_v[i] = &training_images[i * 784];
    printf("done!\n");

    // Augment training data
    if (augmentation_factor > 1) printf("Augmenting training data by a factor of %d...", augmentation_factor);
    fflush(stdout);
    int count = training_count * augmentation_factor;
    uint8_t *labels = augment_labels(training_labels, training_count, augmentation_factor);
    uint8_t *images = augment_images(training_images, training_count, augmentation_factor);
    uint8_t **images_v = (uint8_t**) malloc(sizeof(uint8_t*) * count);
    
    // Every *images_v points to one array of 784 pixels, or one image
    for (int i = 0; i < count; i++)
        images_v[i] = &images[i * 784];
    if (augmentation_factor > 1) printf("done!\n");

    printf("Total training images: %d\n", count);
    
    srand(time(0));

    // Create Neural Network
    node *head;

    head = dl_create(n_inputs, n_layers, sizes);
    if (dl_check(head) < n_layers + 1)
    {
        fprintf(stderr, "Network creation: %d layers found, %d expected.\n", dl_check(head), n_layers + 1);
        return 0;
    }
    dl_print_structure(head);

    // Store average cost over batches and limit how many are processed
    double avg_cost;
    int batches = count / batch_size;
    double *costs;
    costs = (double*) malloc(sizeof(double) * batches);
    for (int i = 0; i < batches; i++)
        costs[i] = 0;

    // Actual training
    matrix input = matrix_create(n_inputs, 1);
    matrix output;
    matrix expected = matrix_create(10, 1);

    // verification
    int correct_answers;
    int choice;
    double confidence = 0;
    char percent[8];
    int percent_len;

    printf("%d epochs in batches of %d pictures.\nLearning rate is %lf.\n", epochs, batch_size, alpha);
    for (int epoch = 0; epoch < epochs; epoch++)
    {    
        shuffle_images(images_v, labels, count);
        avg_cost = 0;
        correct_answers = 0;

        printf("Training epoch %d...", epoch);
        fflush(stdout);
        percent[0] = '\0';
        for (int i = 0; i < batches; i++)
        {
            // Training run progress
            for (int n = 0; percent[n] != '\0'; n++)
                putc('\b', stdout);
            sprintf(percent, "%.2lf%%", (double) i / batches * 100);
            printf("%s", percent);
            fflush(stdout);

            for (int j = 0; j < batch_size; j++)
            {
                // input and expected result creation
                for (int k = 0; k < n_inputs; k++)
                {
                    input.matrix[k][0] = (double) images_v[i * batch_size + j][k];
                }
                expected = matrix_zero(expected);
                expected.matrix[labels[i * batch_size + j]][0] = 1;

                // Process and store adjustments
                output = dl_process(head, input);
                dl_backwards_pass(head, expected, alpha);
                costs[i] += dl_cost(output, expected);

                // dbg: testing correctness
                // verification
                choice = 0;
                for (int m = 0; m < output.height; m++)
                {
                    if (output.matrix[m][0] > confidence)
                    {
                        confidence = output.matrix[m][0];
                        choice = m;
                    }
                }
                confidence = 0;
        
                // counting
                correct_answers += (choice == labels[i * batch_size + j]);

                matrix_free(output);
            }
            costs[i] /= batch_size;
            dl_adjust(head);
        }
        
        for (int l = 0; l < batches; l++)
        {
            avg_cost += costs[l];
        }
        avg_cost /= batches;
        printf("\rAvg cost over epoch %d: %.4lf\tAcc: %.2lf%%\n", epoch, avg_cost, (double) correct_answers / (batch_size * batches) * 100);
    }

    // Save model
    dl_dump(head, filename);

    // Test model
    test_model(head);
    
    // Clean up
    matrix_free(input);
    matrix_free(expected);

    dl_free(head);

    free(sizes);
    free(training_labels);
    free(training_images);
    free(training_images_v);
    free(labels);
    free(images);
    free(images_v);
    return 0;
}