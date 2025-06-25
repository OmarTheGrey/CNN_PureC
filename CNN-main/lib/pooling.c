/*
 * pooling.c — 2×2 max-pool layer implementation
 * ----------------------------------------------
 * Down-samples each feature map by a factor of two
 * via max-pooling. Simple and fast; no trainable
 * parameters.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "pooling.h"

/*
 * tabMax()
 * Utility: returns the maximum value of an array
 * of length `size`. Used by the pooling layer.
 */
double tabMax(double* tab, int size) {
    double max = tab[0];
    for (int i=0; i<size; i++) {
        if (tab[i] > max) {
            max = tab[i];
        }
    }
    return max;
}

/*
 * poolingForward()
 * Performs 2×2 max-pooling on each filter channel.
 * The input is a flattened convolution grid where
 * every cell holds `numFilters` activations.
 * Returns a flat array that interleaves channels:
 *   [c0, c0, …, c1, c1, …]
 */
double* poolingForward(double** input, int width, int height, int numFilters) {
    double** output = malloc(width * height * sizeof(double));
    assert(output != NULL);

    for (int i=0; i<width*height; i++) {
        output[i] = malloc(numFilters * sizeof(double));
        assert(output[i] != NULL);

        for (int k=0; k<numFilters; k++) {
            double* cell = malloc(4 * sizeof(double));
            assert(cell != NULL);

            cell[0] = input[2*i][k];
            cell[1] = input[2*i + 1][k];
            cell[2] = input[2*i + width][k];
            cell[3] = input[2*i + width + 1][k];

            output[i][k] = tabMax(cell, 4);
            free(cell);
        }
    }

    double* flatOutput = malloc(width * height * numFilters * sizeof(double));
    assert(flatOutput != NULL);

    for (int i=0; i<numFilters; i++) {
        for (int j=0; j<width*height; j++) {
            flatOutput[i*width*height+j] = output[j][i];
        }
    }

    for (int i=0; i<width*height; i++) {
        free(output[i]);
    }
    free(output);

    return flatOutput;
}