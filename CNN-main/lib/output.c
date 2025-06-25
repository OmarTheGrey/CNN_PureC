/*
 * output.c — Softmax + metrics helpers
 * ------------------------------------
 * Final activation layer and training metrics.
 * Nothing fancy, just exponential + normalisation
 * and a couple of convenience routines.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "dense.h"

/*
 * softmax()
 * Converts raw logits into a probability distribution.
 */
double* softmax(double* input, int size) {
    double* output = malloc(size * sizeof(double));
    assert(output != NULL);

    double sum = 0.0;
    for (int i=0; i<size; i++) {
        sum += exp(input[i]);
    }

    for (int i=0; i<size; i++) {
        output[i] = exp(input[i]) / sum;
    }
    return output;
}

/*
 * loss()
 * Negative log-likelihood for the correct class.
 */
double loss(double* probs, int label) {
    return -log(probs[label]);
}

/*
 * accuracy()
 * Returns 1 if argmax(probs) equals the ground-truth label.
 */
int accuracy(double* probs, int label, int size) {
    double max = probs[0];
    int index = 0;
    for (int i=0; i<size; i++) {
        if (probs[i] > max) {
            max = probs[i];
            index = i;
        }
    }
    
    if (index == label) return 1;
    else return 0;
}