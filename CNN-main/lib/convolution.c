/*
 * convolution.c — Convolution layer implementation
 * -------------------------------------------------
 * Handles filter initialisation (He), forward pass grid
 * construction and the actual 2-D convolution used in our
 * toy MNIST CNN. Written for learning purposes, so the code
 * trades some performance for clarity.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "convolution.h"

/*
 * convBoxMuller()
 * Returns one sample from a standard normal distribution
 * using the Box-Muller transform. Used for He-style weight
 * initialisation of the filters.
 */
double convBoxMuller() {
    static int hasSpare = 0;
    static double spare;
    
    if (hasSpare) {
        hasSpare = 0;
        return spare;
    }

    hasSpare = 1;
    double u, v, s;
    do {
        u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return u * s;
}

/*
 * initConvLayer()
 * Allocates a convolutional layer structure and initialises
 * `numFilters` square filters of size `filterSize`×`filterSize`
 * with He-initialised Gaussian noise.
 */
ConvLayer* initConvLayer(int numFilters, int filterSize) {
    ConvLayer* layer = malloc(sizeof(ConvLayer));
    assert(layer != NULL);

    layer->numFilters = numFilters;
    layer->filterSize = filterSize;
    layer->filters = malloc(numFilters * sizeof(double**));
    assert(layer->filters != NULL);

    for (int i=0; i<numFilters; i++) {
        layer->filters[i] = malloc(filterSize * sizeof(double*));
        assert(layer->filters[i] != NULL);
        for (int j=0; j<filterSize; j++) {
            layer->filters[i][j] = malloc(filterSize * sizeof(double));
            for (int k=0; k<filterSize; k++) {
                double heInit = convBoxMuller() * sqrt(2.0 / ((double)filterSize * (double)filterSize));
                layer->filters[i][j][k] = heInit;
            }
        }
    }
    return layer;
}

/*
 * freeConvLayer()
 * Frees all heap allocations belonging to a ConvLayer.
 */
void freeConvLayer(ConvLayer* layer) {
    for (int i=0; i<layer->numFilters; i++) {
        for (int j=0; j<layer->filterSize; j++) {
            free(layer->filters[i][j]);
        }
        free(layer->filters[i]);
    }
    free(layer->filters);
    free(layer);
}

/*
 * convolutionGrid()
 * Slides a `divisor`×`divisor` window across the input image
 * and flattens each patch into a 1-D cell. Returns an array
 * of those cells so that the actual convolution becomes a
 * dot-product.
 */
double** convolutionGrid(double** image, int width, int height, int divisor) {
    double** grid = malloc((width - (divisor-1)) * (height - (divisor-1)) * sizeof(double*));
    assert(grid != NULL);

    for (int i=0; i<(width - (divisor-1)); i++) {
        for (int j=0; j<(height - (divisor-1)); j++) {
            double* cell = malloc(divisor * divisor * sizeof(double));
            assert(cell != NULL);

            for (int k=0; k<divisor; k++) {
                for (int l=0; l<divisor; l++) {
                    cell[k * divisor + l] = image[i + k][j + l];
                }
            }

            grid[i * (height - (divisor-1)) + j] = cell;
        }
    }
    return grid;
}

/*
 * convolution()
 * Computes the dot-product between a single filter and one
 * flattened image cell.
 */
double convolution(double** filter, double* cell) {
    double sum = 0.0;
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            sum += cell[i * 3 + j] * filter[i][j];
        }
    }
    return sum;
}

/*
 * convolutionForward()
 * Produces the convolved feature maps for all filters.
 * Output is a `(w-div+1)×(h-div+1)` grid where each entry
 * is an array of `numFilters` activations.
 */
double** convolutionForward(ConvLayer* convLayer, double** image, int width, int height, int divisor) {
    double** grid = convolutionGrid(image, width, height, divisor);
    double** output = malloc((width - (divisor-1)) * (height - (divisor-1)) * sizeof(double*));
    assert(output != NULL);

    for (int i=0; i<(width - (divisor-1)); i++) {
        for (int j=0; j<(height - (divisor-1)); j++) {
            output[i * (height - (divisor-1)) + j] = malloc(convLayer->numFilters * sizeof(double));
            assert(output[i * (height - (divisor-1)) + j] != NULL);
            for (int k=0; k<convLayer->numFilters; k++) {
                output[i * (height - (divisor-1)) + j][k] = convolution(convLayer->filters[k], grid[i * (height - (divisor-1)) + j]);
            }
        }
    }
    return output;
}