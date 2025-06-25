/*
 * pooling.h — interface for 2×2 max-pool layer
 * --------------------------------------------
 * Keeps it ultra-simple: only a single forward
 * routine, no trainable weights.
 */

#ifndef POOLING_H
#define POOLING_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

double* poolingForward(double** input, int width, int height, int numFilters);

#endif